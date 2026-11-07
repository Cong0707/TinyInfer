
import time, torch
from tinyinfer.kernels import native
from tinyinfer.deepseek.hc import _hc_post_ref

def bench(fn, R=500, w=100):
    for _ in range(w): fn()
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(R): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t0)/R*1e6

# decode 的实际形状：M=1, hc=4, D=3072
for (M, hc, D) in [(1,4,3072), (1,4,7168), (8,4,3072)]:
    x    = torch.randn(M, D, dtype=torch.bfloat16, device="cuda")
    res  = torch.randn(M, hc, D, dtype=torch.bfloat16, device="cuda")
    post = torch.randn(M, hc, dtype=torch.float32, device="cuda")
    comb = torch.randn(M, hc, hc, dtype=torch.float32, device="cuda")
    out = torch.empty((M, hc, D), dtype=torch.bfloat16, device="cuda")
    n = M*hc*D
    kern = native.kernel("hc", "hc_post_bf")
    tn = bench(lambda: native.launch(kern, ((n+255)//256,), (256,),
                                     [x, res, post, comb, out, M, hc, D]))
    tt = bench(lambda: _hc_post_ref(x, res, post, comb))
    print(f"M={M} D={D:<5}  cubin {tn:6.2f}us   torch {tt:7.2f}us   {tt/tn:.1f}x")
