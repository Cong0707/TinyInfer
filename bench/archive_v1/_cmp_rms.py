
import sys
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
import torch
import vllm._C_stable_libtorch  # noqa
from tinyinfer.nn import rms_norm as mine

dev = torch.device("cuda")
def bench(fn, warmup=10, iters=100):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1000.0

print(f"{'M':>6} {'K':>6} | {'mine us':>9} {'vllm us':>9} | ratio")
for M, K in ((1, 5120), (8, 5120), (4096, 4096)):
    x = torch.randn(M, K, dtype=torch.bfloat16, device=dev)
    w = torch.randn(K, dtype=torch.bfloat16, device=dev)
    out = torch.empty_like(x)
    t_mine = bench(lambda: mine(x, w, 1e-6))
    t_vllm = bench(lambda: torch.ops._C.rms_norm(out, x, w, 1e-6))
    print(f"{M:>6} {K:>6} | {t_mine:>9.2f} {t_vllm:>9.2f} | {t_mine/t_vllm:>5.2f}x")
