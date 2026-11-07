
import os, time, torch
import tinyinfer._C_stable_libtorch  # noqa
from tinyinfer import gemv_fp4_marlin as GM
dev="cuda"; N,K = 2048, 7168
w = torch.randint(0, 256, (N, K//2), dtype=torch.uint8, device=dev)
s = torch.full((N, K//32), 127, dtype=torch.uint8, device=dev)
x = torch.randn(1, K, dtype=torch.bfloat16, device=dev)
GM.gemv_fp4_fast(x, w, s); torch.cuda.synchronize()
def bench(fn, R=200, w_=50):
    for _ in range(w_): fn()
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(R): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t0)/R*1e6
print("mode:", "PYD" if os.environ.get("TI_MARLIN_PYD")=="1" else "NATIVE")
for i in range(3):
    print(f"  {bench(lambda: GM.gemv_fp4_fast(x, w, s)):.1f} us   (N={N} K={K})")
