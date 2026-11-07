import sys
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
import torch
import tinyinfer.torch_compat  # noqa
from tinyinfer import native

def bench(fn, n=25, warm=6):
    with torch.inference_mode():
        for _ in range(warm): fn()
        torch.cuda.synchronize()
        e0 = torch.cuda.Event(True); e1 = torch.cuda.Event(True)
        e0.record()
        for _ in range(n): fn()
        e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1)/n

# correctness at small K
N, K = 256, 512
words = torch.randint(0, 2**31 - 1, (N, K//8), dtype=torch.int64)
w = words.to(torch.int32).to("cuda")
wsm = torch.full((N, K//32), 127, dtype=torch.uint8, device="cuda")
torch.manual_seed(0)
x = torch.randn(K, dtype=torch.bfloat16, device="cuda")
u = words & 0xFFFFFFFF
codes = torch.zeros((N, K), dtype=torch.int64)
for i in range(8):
    codes[:, i::8] = (u >> (4 * i)) & 0xF
e2m1 = torch.tensor([0,0.5,1,1.5,2,3,4,6, 0,-0.5,-1,-1.5,-2,-3,-4,-6])
ref = e2m1[codes] @ x.cpu().float()
y = torch.zeros(N, dtype=torch.bfloat16, device="cuda")
native.launch(native.kernel("marlin_fp4","e2m1_v3"), ((N + 63)//64,), (128,), [x, w, wsm, y, N, K])
torch.cuda.synchronize()
d = (y.cpu().float() - ref).abs()
print(f"correctness (K=512): max {d.max().item():.4f} rel {(d.max()/ref.abs().max()).item():.2e}")

# speed at real K
N, K = 16384, 5120
w2 = torch.randint(0, 2**31 - 1, (N, K//8), dtype=torch.int32, device="cuda")
ws2 = torch.full((N, K//32), 127, dtype=torch.uint8, device="cuda")
x2 = torch.randn(K, dtype=torch.bfloat16, device="cuda")
y2 = torch.zeros(N, dtype=torch.bfloat16, device="cuda")
ms = bench(lambda: native.launch(native.kernel("marlin_fp4","e2m1_v3"), ((N + 63)//64,), (128,),
                                 [x2, w2, ws2, y2, N, K]))
by = N*K*0.5
print(f"e2m1_v3: {ms*1000:8.1f} us -> {by/(ms/1000)/1e9:6.0f} GB/s")
print("  (tilelang 41, scalar 62, pure-read 212)")
