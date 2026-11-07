
import sys
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
import torch
from tinyinfer.moe import Expert

dev = torch.device("cuda")
N, K = 16384, 5120
def pack(r, k):
    return torch.randint(0, 256, (r, k // 2), dtype=torch.uint8, device=dev)
def sca(r, k):
    return torch.randint(120, 128, (r, k // 32), dtype=torch.uint8, device=dev)

e = Expert(None, None, None, swiglu_limit=0.0)
# w1/w3: [N, K/2]   w2: [K, N/2]  (down projection)
e.fp4 = (pack(N, K), sca(N, K), pack(K, N), sca(K, N), pack(N, K), sca(N, K))

x = torch.randn(1, K, dtype=torch.bfloat16, device=dev)
y = e.forward_fp4(x)
torch.cuda.synchronize()
print("forward_fp4 out", tuple(y.shape), y.dtype, "finite", bool(torch.isfinite(y).all()))

def bench(fn, warmup=5, iters=30):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(True); e2 = torch.cuda.Event(True)
    s.record()
    for _ in range(iters): fn()
    e2.record(); torch.cuda.synchronize()
    return s.elapsed_time(e2) / iters * 1000.0

t = bench(lambda: e.forward_fp4(x))
print(f"one expert FFN (3 GEMVs) M=1: {t:.1f} us  -> per GEMV {t/3:.1f} us")

x4 = torch.randn(4, K, dtype=torch.bfloat16, device=dev)
y4 = e.forward_fp4(x4)
print("M=4 out", tuple(y4.shape), "finite", bool(torch.isfinite(y4).all()))
