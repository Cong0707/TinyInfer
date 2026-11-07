
import sys
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
import torch
import tinyinfer._C_stable_libtorch  # noqa

dev = torch.device("cuda")
def bench(fn, warmup=5, iters=30):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1000.0

f8 = torch.float8_e4m3fn
print("== cutlass_scaled_mm (fp8 W8A8, per-token x per-channel) ==")
for M, N, K in ((1, 16384, 5120), (8, 16384, 5120), (128, 16384, 5120), (4096, 4096, 4096)):
    a = (torch.randn(M, K, device=dev) / 4).to(f8)
    # weight is stored [N, K] row-major; the kernel wants [K, N] column-major
    b = (torch.randn(N, K, device=dev) / 4).to(f8).T
    asc = torch.rand(M, 1, device=dev) + 0.5
    bsc = torch.rand(1, N, device=dev) + 0.5
    out = torch.empty(M, N, dtype=torch.bfloat16, device=dev)
    try:
        t = bench(lambda: torch.ops._C.cutlass_scaled_mm(out, a, b, asc, bsc, None))
        print(f"  M={M:5d} N={N:5d} K={K:5d}: {t:8.2f} us   {N*K/(t*1e-6)/1e9:6.1f} GB/s")
    except Exception as ex:
        print(f"  M={M} FAIL {str(ex)[:200]}")
