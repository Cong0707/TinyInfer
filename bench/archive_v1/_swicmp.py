
import time, torch
from tinyinfer.kernels.tilelang.quant.swiglu_forward_kernel import swiglu_forward
from tinyinfer.vllm_ops import clamp_swiglu

def bench(fn, R=300, w=80):
    for _ in range(w): fn()
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(R): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t0)/R*1e6

LIMIT = 10.0
for (T, D) in [(1, 288), (1, 2304), (8, 2304)]:
    x = torch.randn(T, 2*D, device="cuda", dtype=torch.bfloat16)
    try:
        a = swiglu_forward(x, "bf16", clamp_value=LIMIT)
        torch.cuda.synchronize()
    except Exception as e:
        print(f"T={T} D={D} tilelang FAILED: {str(e)[:90]}"); continue
    b = clamp_swiglu(x, LIMIT)
    torch.cuda.synchronize()
    d = float((a.float()-b.float()).abs().max())
    ta = bench(lambda: swiglu_forward(x, "bf16", clamp_value=LIMIT))
    tb = bench(lambda: clamp_swiglu(x, LIMIT))
    print(f"T={T:<3} D={D:<5}  tilelang {ta:6.2f}us   cuda {tb:6.2f}us   maxdiff {d:.2e}")
