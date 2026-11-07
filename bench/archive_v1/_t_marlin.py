
import sys, time
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
import torch
from tinyinfer.gemv_fp4_marlin import gemv_fp4_fast, available, _prepared

print("available:", available())
dev = torch.device("cuda")
N, K = 16384, 5120
w_fp4 = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=dev)
w_scale = torch.randint(120, 128, (N, K // 32), dtype=torch.uint8, device=dev)
x = torch.randn(1, K, dtype=torch.bfloat16, device=dev)

t0 = time.perf_counter()
y = gemv_fp4_fast(x, w_fp4, w_scale)
torch.cuda.synchronize()
print(f"first call (incl. repack): {(time.perf_counter()-t0)*1000:.1f} ms")
print("out", tuple(y.shape), y.dtype, "finite:", bool(torch.isfinite(y).all()), "absmean", y.abs().mean().item())

def bench(fn, warmup=5, iters=30):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1000.0

nbytes = N * K // 2
t = bench(lambda: gemv_fp4_fast(x, w_fp4, w_scale))
print(f"marlin via tinyinfer M=1: {t:8.1f} us   {nbytes/(t*1e-6)/1e9:6.1f} GB/s")

# steady-state confirm
t2 = bench(lambda: gemv_fp4_fast(x, w_fp4, w_scale), warmup=20, iters=100)
print(f"repeat                  : {t2:8.1f} us")
