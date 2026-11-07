
import torch
import tinyinfer._C_stable_libtorch  # noqa
from tinyinfer import gemv_fp4_marlin as GM

dev = "cuda"
N, K = 16384, 5120
w = torch.randint(0, 256, (N, K//2), dtype=torch.uint8, device=dev)
s = torch.full((N, K//32), 127, dtype=torch.uint8, device=dev)
x = torch.randn(1, K, dtype=torch.bfloat16, device=dev)
GM.gemv_fp4_fast(x, w, s)
torch.cuda.synchronize()

from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    GM.gemv_fp4_fast(x, w, s)
    torch.cuda.synchronize()

for e in prof.key_averages():
    if "Marlin" in e.key or "marlin" in e.key:
        print("KERNEL:", e.key)
        print("count:", e.count)
