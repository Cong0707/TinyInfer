
import torch, time
import tinyinfer._C_stable_libtorch  # noqa
from tinyinfer import native
dev="cuda"
N=2048; M=1
x = torch.randn(M, 2*N, dtype=torch.bfloat16, device=dev)
out = torch.empty(M, N, dtype=torch.bfloat16, device=dev)
blk = min(N//8, 1024)
L = native.bind("vllm_silu_and_mul","vllm_silu_and_mul_clamp_bf16",(M,),(blk,),"TTif")

# 大量重复同一个 kernel，用 nsys 式的差分：跑 1 次 vs 跑 100 次的总时间差
def total(fn, n):
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t0)*1e6

for fn, name in ((lambda: L(out,x,N,10.0),"ours"), (lambda: torch.ops._C.silu_and_mul_with_clamp(out,x,10.0),"_C")):
    for _ in range(500): fn()
    torch.cuda.synchronize()
    t1 = total(fn, 100); t2 = total(fn, 1100)
    per = (t2-t1)/1000
    print(f"{name:6} per-call {per:.2f} us")
