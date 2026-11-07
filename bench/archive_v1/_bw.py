
import time, torch
import tinyinfer._C_stable_libtorch  # noqa
def bench(fn, R=50, w=20):
    for _ in range(w): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(R): fn()
    torch.cuda.synchronize()
    return (time.perf_counter()-t0)/R*1e6

dev = "cuda"
# 纯拷读带宽
n = 42 * 1024 * 1024 // 4
src = torch.empty(n, dtype=torch.float32, device=dev)
dst = torch.empty(n, dtype=torch.float32, device=dev)
us = bench(lambda: dst.copy_(src))
print(f"copy 42MB: {us:.1f} us -> {2*42/1024/(us/1e6):.0f} GB/s (read+write)")

# 只读求和
us2 = bench(lambda: src.sum())
print(f"sum  42MB: {us2:.1f} us -> {42/1024/(us2/1e6):.0f} GB/s (read)")

from tinyinfer import gemv_fp4_marlin
N, K = 16384, 5120
w = torch.randint(0, 256, (N, K//2), dtype=torch.uint8, device=dev)
s = torch.full((N, K//32), 127, dtype=torch.uint8, device=dev)
x = torch.randn(1, K, dtype=torch.bfloat16, device=dev)
us3 = bench(lambda: gemv_fp4_marlin.gemv_fp4_fast(x, w, s))
print(f"marlin  42MB: {us3:.1f} us -> {N*K/2/1e6/us3*1e3:.0f} GB/s")
