
import time, torch
import tinyinfer._C_stable_libtorch  # noqa
import tinyinfer._moe_C_stable_libtorch  # noqa
from tinyinfer import gemv_fp4_marlin as GM

dev = "cuda"
N, K = 16384, 5120
w4 = torch.randint(0, 256, (N, K//2), dtype=torch.uint8, device=dev)
s4 = torch.full((N, K//32), 127, dtype=torch.uint8, device=dev)
x1 = torch.randn(1, K, dtype=torch.bfloat16, device=dev)

def bench(fn, R=200, w=50):
    for _ in range(w): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(R): fn()
    torch.cuda.synchronize()
    return (time.perf_counter()-t0)/R*1e6

GM.gemv_fp4_fast(x1, w4, s4)
# 多次采样看稳定性
vals = [bench(lambda: GM.gemv_fp4_fast(x1, w4, s4)) for _ in range(5)]
print("marlin us:", [f"{v:.1f}" for v in vals])
print(f"  中位 {sorted(vals)[2]:.1f} us  {N*K/2/1e6/sorted(vals)[2]*1e3:.0f} GB/s")
