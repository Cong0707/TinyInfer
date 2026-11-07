
import time, torch
from tinyinfer.kernels.tilelang.moe.topk_gate_kernel import topk_gate
from tinyinfer.vllm_ops import gate_topk
import inspect
print("gate_topk sig:", str(inspect.signature(gate_topk))[:130])

def bench(fn, R=300, w=80):
    for _ in range(w): fn()
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(R): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t0)/R*1e6

for T in (1, 8):
    x = torch.randn(T, 384, device="cuda", dtype=torch.float32)
    try:
        a = topk_gate(x, 6); torch.cuda.synchronize()
        ta = bench(lambda: topk_gate(x, 6))
    except Exception as e:
        print(f"T={T} tilelang FAILED: {str(e)[:80]}"); continue
    try:
        b = gate_topk(x); torch.cuda.synchronize()
        tb = bench(lambda: gate_topk(x))
        print(f"T={T} tilelang {ta:6.2f}us  cuda {tb:6.2f}us   tl_shape={tuple(a.shape)}")
    except Exception as e:
        print(f"T={T} tilelang {ta:6.2f}us  cuda FAILED: {str(e)[:100]}")
