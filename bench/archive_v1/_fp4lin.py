
import time, torch
import tinyinfer._C_stable_libtorch  # noqa
from tinyinfer.gemm import fp4_linear

dev = "cuda"
N, K = 16384, 5120
torch.manual_seed(0)
w = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=dev)
s = torch.full((N, K // 32), 127, dtype=torch.uint8, device=dev)
x = torch.randn(1, K, dtype=torch.bfloat16, device=dev)

o = fp4_linear(x, w, s); torch.cuda.synchronize()
print("out:", tuple(o.shape), o.dtype)
for _ in range(30): fp4_linear(x, w, s)
torch.cuda.synchronize()
t0 = time.perf_counter(); R = 100
for _ in range(R): fp4_linear(x, w, s)
torch.cuda.synchronize()
us = (time.perf_counter()-t0)/R*1e6
print(f"fp4_linear: {us:.1f} us  {N*K/2/1e6/us*1e3:.1f} GB/s")
