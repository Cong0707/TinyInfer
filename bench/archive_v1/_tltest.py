
import time, torch
import tilelang
print("tilelang:", tilelang.__version__)

from tinyinfer.kernels.tilelang.moe.topk_gate_kernel import topk_gate
from tinyinfer.kernels.tilelang.quant.swiglu_forward_kernel import swiglu_forward

def bench(fn, R=200, w=50):
    for _ in range(w): fn()
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(R): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t0)/R*1e6

# --- topk_gate: [T,384] -> top6 ---
x = torch.randn(1, 384, device="cuda", dtype=torch.float32)
try:
    r = topk_gate(x, 6); torch.cuda.synchronize()
    print("topk_gate ok ->", tuple(r.shape) if hasattr(r,'shape') else type(r))
    print(f"  tilelang: {bench(lambda: topk_gate(x, 6)):.2f} us")
except Exception as e:
    print("topk_gate FAILED:", type(e).__name__, str(e)[:200])

# --- swiglu: gate/up -> mul ---
T, D = 1, 288
g = torch.randn(T, D, device="cuda", dtype=torch.bfloat16)
u = torch.randn(T, D, device="cuda", dtype=torch.bfloat16)
try:
    r = swiglu_forward(g, u, clamp_value=10.0); torch.cuda.synchronize()
    print("swiglu ok ->", tuple(r.shape), r.dtype)
    print(f"  tilelang: {bench(lambda: swiglu_forward(g, u, clamp_value=10.0)):.2f} us")
except Exception as e:
    print("swiglu FAILED:", type(e).__name__, str(e)[:200])
