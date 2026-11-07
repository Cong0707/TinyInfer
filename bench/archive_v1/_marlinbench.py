import sys
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
import torch
import tinyinfer.torch_compat  # noqa: F401
from tinyinfer import native

N, K = 4096, 5120
# weights as uint32 words: 8 codes each
w = torch.randint(0, 2**31, (N, K//8), dtype=torch.int32, device="cuda").view(torch.uint32) if hasattr(torch,'uint32') else torch.randint(0, 2**31, (N, K//8), dtype=torch.int32, device="cuda")
wsc = torch.full((N, K//32), 1.0, dtype=torch.float16, device="cuda")
x = torch.randn(K, dtype=torch.bfloat16, device="cuda") * 0.1
asc = torch.ones(K//32, dtype=torch.float32, device="cuda")
y = torch.zeros(N, dtype=torch.bfloat16, device="cuda")

def bench(fn, n=30, warm=8):
    with torch.inference_mode():
        for _ in range(warm): fn()
        torch.cuda.synchronize()
        e0 = torch.cuda.Event(True); e1 = torch.cuda.Event(True)
        e0.record()
        for _ in range(n): fn()
        e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1)/n

grid = ((N + 15)//16,)
try:
    native.launch(native.kernel("marlin_fp4","marlin_fp4_gemv"), grid, (128,),
                  [x, asc, w.view(torch.int32), wsc, y, N, K], shared=0)
    torch.cuda.synchronize()
    ms = bench(lambda: native.launch(native.kernel("marlin_fp4","marlin_fp4_gemv"), grid, (128,),
                                     [x, asc, w.view(torch.int32), wsc, y, N, K]))
    by = N*K*0.5
    print(f"marlin mma: {ms*1000:8.1f} us -> {by/(ms/1000)/1e9:6.0f} GB/s")
    print(f"  (scalar ALU was 62 GB/s, tilelang 41, pure read 212)")
    print(f"  y nonzero: {int((y != 0).sum())}/{N}")
except Exception as e:
    print("failed:", type(e).__name__, str(e)[:300])
