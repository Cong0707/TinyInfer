import sys
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
import torch
import tinyinfer.torch_compat  # noqa
from tinyinfer import native

def ref_check(N, K, x, words_np):
    u = words_np & 0xFFFFFFFF
    codes = torch.zeros((N, K), dtype=torch.int64)
    for i in range(8):
        codes[:, i::8] = (u >> (4 * i)) & 0xF
    e2m1 = torch.tensor([0,0.5,1,1.5,2,3,4,6, 0,-0.5,-1,-1.5,-2,-3,-4,-6])
    vals = e2m1[codes]
    return (vals @ x.cpu().float())

N, K = 256, 512
words = torch.randint(0, 2**31 - 1, (N, K//8), dtype=torch.int64)
w = words.to(torch.int32).to("cuda")
ws = torch.full((N, K//32), 127, dtype=torch.uint8, device="cuda")
torch.manual_seed(0)
x = torch.randn(K, dtype=torch.bfloat16, device="cuda")

y = torch.zeros(N, dtype=torch.bfloat16, device="cuda")
native.launch(native.kernel("marlin_fp4","e2m1_gemv"), ((N + 63)//64,), (128,), [x, w, ws, y, N, K])
torch.cuda.synchronize()
ref = ref_check(N, K, x, words)
d = (y.cpu().float() - ref).abs()
print(f"max {d.max().item():.4f}  mean {d.mean().item():.4f}")
print(f"  y  [0:6]: {[round(v,3) for v in y[:6].tolist()]}")
print(f"  ref[0:6]: {[round(v,3) for v in ref[:6].tolist()]}")
print(f"  rel: {(d.max()/ref.abs().max()).item():.2e}")
