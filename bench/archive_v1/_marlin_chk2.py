
import time, torch
import tinyinfer._C_stable_libtorch  # noqa
import tinyinfer._moe_C_stable_libtorch  # noqa
from tinyinfer import gemv_fp4_marlin

dev = "cuda"
N, K = 16384, 5120
torch.manual_seed(0)
w = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=dev)
s = torch.full((N, K // 32), 127, dtype=torch.uint8, device=dev)
x = torch.randn(1, K, dtype=torch.bfloat16, device=dev)

t0 = time.perf_counter()
o = gemv_fp4_marlin.gemv_fp4_fast(x, w, s)
torch.cuda.synchronize()
print(f"first (includes repack): {(time.perf_counter()-t0)*1e3:.1f} ms")

for rnd in range(3):
    for _ in range(20): gemv_fp4_marlin.gemv_fp4_fast(x, w, s)
    torch.cuda.synchronize()
    t0 = time.perf_counter(); R = 100
    for _ in range(R): gemv_fp4_marlin.gemv_fp4_fast(x, w, s)
    torch.cuda.synchronize()
    us = (time.perf_counter() - t0) / R * 1e6
    print(f"round{rnd}: {us:.1f} us  {N*K/2/1e6/us*1e3:.1f} GB/s")
