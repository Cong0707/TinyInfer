"""Measure the tilelang fp4 kernel that the engine actually uses."""
import sys, time
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
import torch
import tinyinfer.torch_compat  # noqa: F401
from tinyinfer import gemv_fp4

N, K = 16384, 5120
MB = 2**20
w = torch.randint(0, 256, (N, K//2), dtype=torch.uint8, device="cuda")
ws = torch.ones(N, K//32, dtype=torch.float32, device="cuda")
a = torch.randn(K, dtype=torch.bfloat16, device="cuda")
asc = torch.ones(K//32, dtype=torch.float32, device="cuda")

t0 = time.perf_counter()
y = gemv_fp4.gemv_fp4(a, w, ws)
print(f"first call (compiles tilelang): {time.perf_counter()-t0:.1f}s", flush=True)
torch.cuda.synchronize()

def bench(fn, n=50, warm=10):
    with torch.inference_mode():
        for _ in range(warm): fn()
        torch.cuda.synchronize()
        e0 = torch.cuda.Event(True); e1 = torch.cuda.Event(True)
        e0.record()
        for _ in range(n): fn()
        e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1)/n

ms = bench(lambda: gemv_fp4.gemv_fp4(a, w, ws))
by = N*K*0.5
print(f"tilelang fp4 gemv: {ms*1000:8.1f} us -> {by/(ms/1000)/1e9:6.0f} GB/s", flush=True)
print(f"  (dataset {by/MB:.0f} MB, pure-read ceiling measured at 212 GB/s)", flush=True)
