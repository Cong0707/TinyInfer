
import time, torch, torch.nn.functional as F
import tinyinfer._C_stable_libtorch  # noqa
from tinyinfer import native
dev="cuda"

def bench(fn, R=3000, w_=300):
    for _ in range(w_): fn()
    torch.cuda.synchronize()
    t0=time.perf_counter()
    for _ in range(R): fn()
    torch.cuda.synchronize()
    return (time.perf_counter()-t0)/R*1e6

def gpu_time(fn, R=2000):
    for _ in range(200): fn()
    torch.cuda.synchronize()
    s,e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(R): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e)/R*1000

for (M, N) in [(1, 2048), (6, 2048), (1, 4096), (64, 2048)]:
    x = torch.randn(M, 2*N, dtype=torch.bfloat16, device=dev)
    ref = torch.empty(M, N, dtype=torch.bfloat16, device=dev)
    mine = torch.empty_like(ref)
    torch.ops._C.silu_and_mul_with_clamp(ref, x, 10.0)
    blk = min(N // 8, 1024)
    L = native.bind("vllm_silu_and_mul","vllm_silu_and_mul_clamp_bf16",(M,),(blk,),"TTif")
    L(mine, x, N, 10.0); torch.cuda.synchronize()
    diff = float((ref-mine).abs().max())
    u_ref = bench(lambda: torch.ops._C.silu_and_mul_with_clamp(ref, x, 10.0))
    u_min = bench(lambda: L(mine, x, N, 10.0))
    g_ref = gpu_time(lambda: torch.ops._C.silu_and_mul_with_clamp(ref, x, 10.0))
    g_min = gpu_time(lambda: L(mine, x, N, 10.0))
    print(f"M={M:<3} N={N:<5} diff={diff:.6f}  cpu: _C={u_ref:5.2f} ours={u_min:5.2f}  gpu: _C={g_ref:5.2f} ours={g_min:5.2f} ({g_ref/g_min:.2f}x)")
