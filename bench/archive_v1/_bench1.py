import time, torch
import vllm._C_stable_libtorch  # noqa
dev = "cuda"
def bench(fn, warmup=10, iters=50):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters

# rms_norm
M, K = 4096, 4096
x = torch.randn(M, K, dtype=torch.bfloat16, device=dev)
w = torch.randn(K, dtype=torch.bfloat16, device=dev)
out = torch.empty_like(x)
t = bench(lambda: torch.ops._C.rms_norm(out, x, w, 1e-6))
print(f"rms_norm M={M} K={K}: {t*1000:.1f} us  {M*K*2*2/t/1e6:.0f} GB/s")

# silu_and_mul
M2, N2 = 4096, 4096
xi = torch.randn(M2, 2*N2, dtype=torch.bfloat16, device=dev)
o2 = torch.empty(M2, N2, dtype=torch.bfloat16, device=dev)
t = bench(lambda: torch.ops._C.silu_and_mul(o2, xi))
print(f"silu_and_mul M={M2} N={N2}: {t*1000:.1f} us  {M2*N2*2*3/t/1e6:.0f} GB/s")

# scaled_fp4_quant
xq = torch.randn(4096, 4096, dtype=torch.bfloat16, device=dev)
sq = torch.ones(1, dtype=torch.float32, device=dev) * 1.0
t = bench(lambda: torch.ops._C.scaled_fp4_quant(xq, sq, False))
print(f"scaled_fp4_quant 4096x4096: {t*1000:.1f} us")