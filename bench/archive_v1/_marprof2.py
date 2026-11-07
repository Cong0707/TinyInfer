
import torch
import tinyinfer._C_stable_libtorch  # noqa
from tinyinfer import gemv_fp4_marlin as GM
dev="cuda"
for (N,K) in [(2048,7168),(7168,2048)]:
    w = torch.randint(0, 256, (N, K//2), dtype=torch.uint8, device=dev)
    s = torch.full((N, K//32), 127, dtype=torch.uint8, device=dev)
    x = torch.randn(1, K, dtype=torch.bfloat16, device=dev)
    GM.gemv_fp4_fast(x, w, s); torch.cuda.synchronize()
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        # 强制 pyd
        import os; os.environ["TI_MARLIN_PYD"]="1"
        GM.gemv_fp4_fast(x, w, s)
        torch.cuda.synchronize()
        os.environ.pop("TI_MARLIN_PYD")
    for e in prof.key_averages():
        if "Marlin" in e.key:
            print(f"N={N} K={K}:", e.key[:130])
    del w, s, x
