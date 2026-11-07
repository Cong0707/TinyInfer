
import torch
import tinyinfer._C_stable_libtorch  # noqa
from tinyinfer import gemv_fp4_marlin as GM
dev="cuda"
N,K = 2048, 7168
w = torch.randint(0, 256, (N, K//2), dtype=torch.uint8, device=dev)
s = torch.full((N, K//32), 127, dtype=torch.uint8, device=dev)
GM._CACHE.clear()
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    GM._prepared(w, s)
    torch.cuda.synchronize()
for e in prof.key_averages():
    if e.count and ("repack" in e.key.lower() or "Repack" in e.key):
        print("KERNEL:", e.key[:160])
    elif e.count and e.self_device_time_total > 0:
        print("  other:", e.key[:80], int(e.self_device_time_total))
