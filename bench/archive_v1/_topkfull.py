
import time, torch
import tinyinfer._C_stable_libtorch, tinyinfer._moe_C_stable_libtorch  # noqa
from tinyinfer import native
dev="cuda"; E,K = 384,6
def bench(fn, R=3000, w_=300):
    for _ in range(w_): fn()
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(R): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t0)/R*1e6

print(f"{'T':>5} {'idx mism':>9} {'wdiff':>10} {'_C us':>8} {'ours us':>9} {'speedup':>8}")
for T in (1, 2, 8, 32, 128):
    logits = torch.randn(T, E, dtype=torch.float32, device=dev)
    bias = torch.randn(E, dtype=torch.float32, device=dev)
    wr = torch.empty(T,K,dtype=torch.float32,device=dev); ir = torch.empty(T,K,dtype=torch.int32,device=dev)
    tei = torch.empty(T,K,dtype=torch.int32,device=dev)
    torch.ops._moe_C.topk_softplus_sqrt(wr,ir,tei,logits,True,1.5,bias,None,None,None,None,0)
    wm = torch.empty_like(wr); im = torch.empty_like(ir); sm = torch.empty_like(tei)
    block = 32 * min(32, max(1, T))
    grid = (T*32 + block - 1)//block
    L = native.bind("vllm_topk_softplus_sqrt","vllm_topk_softplus_sqrt_e384_k6",(grid,),(block,),"TTTTTiif")
    L(logits, wm, im, sm, bias, T, 1, 1.5); torch.cuda.synchronize()
    mism = int((ir!=im).sum()); dw = float((wr-wm).abs().max())
    ur = bench(lambda: torch.ops._moe_C.topk_softplus_sqrt(wr,ir,tei,logits,True,1.5,bias,None,None,None,None,0))
    um = bench(lambda: L(logits, wm, im, sm, bias, T, 1, 1.5))
    print(f"{T:>5} {mism:>9} {dw:>10.2e} {ur:>8.2f} {um:>9.2f} {ur/um:>7.2f}x")
