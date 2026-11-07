
import time, torch
import tinyinfer._C_stable_libtorch  # noqa
from tinyinfer import gemv_fp4_marlin as GM
dev = "cuda"
def bench(fn, R=200, w=50):
    for _ in range(w): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(R): fn()
    torch.cuda.synchronize()
    return (time.perf_counter()-t0)/R*1e6
N, K = 16384, 5120
w4 = torch.randint(0, 256, (N, K//2), dtype=torch.uint8, device=dev)
s4 = torch.full((N, K//32), 127, dtype=torch.uint8, device=dev)
x1 = torch.randn(1, K, dtype=torch.bfloat16, device=dev)
GM.gemv_fp4_fast(x1, w4, s4)
for i in range(3):
    us = bench(lambda: GM.gemv_fp4_fast(x1, w4, s4))
    print(f"marlin run{i}: {us:.1f} us  {N*K/2/1e6/us*1e3:.0f} GB/s")
from tinyinfer.gemv_fp4 import gemv_fp4_pq, quantize_act
aq, asc = quantize_act(x1, K)
for i in range(2):
    us = bench(lambda: gemv_fp4_pq(aq[0], asc[0], w4, s4), R=60, w=20)
    print(f"hand  run{i}: {us:.1f} us  {N*K/2/1e6/us*1e3:.0f} GB/s")
