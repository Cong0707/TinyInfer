
import time, torch, torch.nn.functional as F
import tinyinfer._C_stable_libtorch   # noqa
import tinyinfer._moe_C_stable_libtorch  # noqa
from tinyinfer import vllm_ops as V
from tinyinfer import gemv_fp4_marlin as GM

dev = "cuda"
def bench(fn, R=100, w=30):
    for _ in range(w): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(R): fn()
    torch.cuda.synchronize()
    return (time.perf_counter()-t0)/R*1e6

def row(name, old_us, new_us, note=""):
    sp = old_us/new_us if new_us else 0
    print(f"{name:<26} {old_us:>9.2f} {new_us:>9.2f}   {sp:>5.2f}x  {note}")

print(f"{'operator':<26} {'old(us)':>9} {'new(us)':>9}   {'speedup':>6}")
print("-"*72)

# 1) rms_norm
X = torch.randn(1, 4096, dtype=torch.bfloat16, device=dev)
W = torch.randn(4096, dtype=torch.bfloat16, device=dev)
def rms_torch():
    v = X.float().pow(2).mean(-1, keepdim=True)
    return (X.float() * torch.rsqrt(v + 1e-6) * W.float()).to(torch.bfloat16)
row("rms_norm M=1 K=4096", bench(rms_torch), bench(lambda: V.rms_norm(X, W, 1e-6)))

X8 = torch.randn(8, 4096, dtype=torch.bfloat16, device=dev)
def rms_torch8():
    v = X8.float().pow(2).mean(-1, keepdim=True)
    return (X8.float() * torch.rsqrt(v + 1e-6) * W.float()).to(torch.bfloat16)
row("rms_norm M=8 K=4096", bench(rms_torch8), bench(lambda: V.rms_norm(X8, W, 1e-6)))

# 2) silu_and_mul
S = torch.randn(1, 8192, dtype=torch.bfloat16, device=dev)
def sim_torch():
    g, u = S.chunk(2, dim=-1)
    return F.silu(g) * u
row("silu_and_mul M=1 2N=8192", bench(sim_torch), bench(lambda: V.silu_and_mul(S)))

# 3) clamp_swiglu (DeepSeek 实际用的)
def cs_torch():
    g, u = S.chunk(2, dim=-1)
    gf = torch.clamp(g.float(), max=10.0)
    uf = torch.clamp(u.float(), min=-10.0, max=10.0)
    return (F.silu(gf) * uf).to(torch.bfloat16)
row("clamp_swiglu M=1 2N=8192", bench(cs_torch), bench(lambda: V.clamp_swiglu(S, 10.0)))

# 4) gate_topk (router)
E, T = 384, 1
logits = torch.randn(T, E, dtype=torch.float32, device=dev)
bias = torch.randn(E, dtype=torch.float32, device=dev)
def gate_torch():
    s = F.softplus(logits).sqrt()
    lg = s + bias
    order = torch.argsort(lg, dim=-1, descending=True, stable=True)
    idx = order[..., :6]
    w = s.gather(1, idx)
    w = w / (w.sum(dim=-1, keepdim=True) + 1e-20)
    return w * 1.5, idx
row("gate_topk T=1 E=384 k=6", bench(gate_torch), bench(lambda: V.gate_topk(logits, 6, True, 1.5, bias)))

# 5) marlin_gemm vs 手写 GEMV
N, K = 16384, 5120
w4 = torch.randint(0, 256, (N, K//2), dtype=torch.uint8, device=dev)
s4 = torch.full((N, K//32), 127, dtype=torch.uint8, device=dev)
x1 = torch.randn(1, K, dtype=torch.bfloat16, device=dev)
def hand_gemv():
    from tinyinfer.gemv_fp4 import gemv_fp4_pq, quantize_act
    aq, asc = quantize_act(x1, K)
    return gemv_fp4_pq(aq[0], asc[0], w4, s4)
row("fp4 expert GEMV (M=1)", bench(hand_gemv, R=30), bench(lambda: GM.gemv_fp4_fast(x1, w4, s4)))
