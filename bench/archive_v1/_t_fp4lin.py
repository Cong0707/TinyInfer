
import sys
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
import torch
from tinyinfer.gemm import fp4_linear

dev = torch.device("cuda")
N, K = 16384, 5120
w_fp4 = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=dev)
w_scale = torch.randint(120, 128, (N, K // 32), dtype=torch.uint8, device=dev)
x = torch.randn(1, K, dtype=torch.bfloat16, device=dev)

y = fp4_linear(x, w_fp4, w_scale)
torch.cuda.synchronize()
print("shape", tuple(y.shape), y.dtype, "finite", bool(torch.isfinite(y).all()))

def bench(fn, warmup=5, iters=30):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1000.0

t = bench(lambda: fp4_linear(x, w_fp4, w_scale))
print(f"fp4_linear via marlin M=1: {t:.1f} us   {N*K//2/(t*1e-6)/1e9:.1f} GB/s")

# multi-row
x4 = torch.randn(4, K, dtype=torch.bfloat16, device=dev)
y4 = fp4_linear(x4, w_fp4, w_scale)
print("M=4 shape", tuple(y4.shape), "finite", bool(torch.isfinite(y4).all()))
