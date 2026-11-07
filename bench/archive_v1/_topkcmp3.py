
import time, torch
from tinyinfer.kernels.tilelang.moe.topk_gate_kernel import topk_gate
from tinyinfer.vllm_ops import gate_topk

def bench(fn, R=300, w=80):
    for _ in range(w): fn()
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(R): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t0)/R*1e6

for T in (1, 8):
    x = torch.randn(T, 384, device="cuda", dtype=torch.float32)
    a = topk_gate(x, 6); torch.cuda.synchronize()
    b = gate_topk(x, 6, True, 1.5); torch.cuda.synchronize()
    ta = bench(lambda: topk_gate(x, 6))
    tb = bench(lambda: gate_topk(x, 6, True, 1.5))
    # b 是 (weights, indices) 或相似
    bi = b[1] if isinstance(b, (tuple, list)) and len(b) >= 2 else None
    same = "n/a"
    if bi is not None and tuple(bi.shape) == tuple(a.shape):
        same = str(bool((bi.sort(dim=-1).values == a.sort(dim=-1).values).all()))
    print(f"T={T} tilelang {ta:6.2f}us  cuda {tb:6.2f}us  idx_match={same}")
