
import torch
import tinyinfer._C_stable_libtorch  # noqa  (只加载 _C，不加载 _moe_C)
from tinyinfer import gemv_fp4_marlin

dev = "cuda"
N, K = 16384, 5120
torch.manual_seed(0)
w = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=dev)
s = torch.full((N, K // 32), 127, dtype=torch.uint8, device=dev)
x = torch.randn(1, K, dtype=torch.bfloat16, device=dev)
import time
gemv_fp4_marlin.gemv_fp4_fast(x, w, s); torch.cuda.synchronize()
for _ in range(30): gemv_fp4_marlin.gemv_fp4_fast(x, w, s)
torch.cuda.synchronize()
t0 = time.perf_counter(); R = 100
for _ in range(R): gemv_fp4_marlin.gemv_fp4_fast(x, w, s)
torch.cuda.synchronize()
us = (time.perf_counter()-t0)/R*1e6
print(f"only _C: {us:.1f} us  {N*K/2/1e6/us*1e3:.1f} GB/s")
