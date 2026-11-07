import sys
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
import torch
import tinyinfer.torch_compat  # noqa: F401
from tinyinfer import native

N, K = 2048, 5120
w = torch.randint(0, 2**31, (N, K//8), dtype=torch.int32, device="cuda")
x = torch.randn(K, dtype=torch.bfloat16, device="cuda") * 0.1
y = torch.zeros(N, dtype=torch.bfloat16, device="cuda")

def bench(fn, n=20, warm=5):
    with torch.inference_mode():
        for _ in range(warm): fn()
        torch.cuda.synchronize()
        e0 = torch.cuda.Event(True); e1 = torch.cuda.Event(True)
        e0.record()
        for _ in range(n): fn()
        e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1)/n

grid = ((N + 63)//64,)
try:
    native.launch(native.kernel("marlin_fp4","marlin_gemv"), grid, (128,),
                  [x, w, y, N, K])
    torch.cuda.synchronize()
    print("ok, nonzero:", int((y != 0).sum()), "/", N)
    ms = bench(lambda: native.launch(native.kernel("marlin_fp4","marlin_gemv"), grid, (128,),
                                     [x, w, y, N, K]))
    by = N*K*0.5
    print(f"marlin mma: {ms*1000:8.1f} us -> {by/(ms/1000)/1e9:6.0f} GB/s")
    print("  (scalar 62, tilelang 41, pure read 212)")
except Exception as e:
    print("failed:", type(e).__name__, str(e)[:250])
