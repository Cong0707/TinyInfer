
import time, torch
import tinyinfer._C_stable_libtorch, tinyinfer._moe_C_stable_libtorch  # noqa
from tinyinfer import native
dev="cuda"
E, K = 384, 6

def bench(fn, R=3000, w_=300):
    for _ in range(w_): fn()
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(R): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t0)/R*1e6

L = native.bind("vllm_topk_softplus_sqrt","vllm_topk_softplus_sqrt_e384_k6",(1,),(32,),"TTTTTif f".replace(" ",""))

for T in (1, 4, 32, 128):
    logits = torch.randn(T, E, dtype=torch.float32, device=dev)
    bias = torch.randn(E, dtype=torch.float32, device=dev)
    w_ref = torch.empty(T, K, dtype=torch.float32, device=dev)
    i_ref = torch.empty(T, K, dtype=torch.int32, device=dev)
    tei = torch.empty(T, K, dtype=torch.int32, device=dev)
    torch.ops._moe_C.topk_softplus_sqrt(w_ref, i_ref, tei, logits, True, 1.5, bias,
                                        None, None, None, None, 0)
    w_m = torch.empty_like(w_ref); i_m = torch.empty_like(i_ref); s_m = torch.empty_like(tei)
    warps = min(1024, max(32, T*32)); warps = (warps//32)*32
    L2 = native.bind("vllm_topk_softplus_sqrt","vllm_topk_softplus_sqrt_e384_k6",( (T*32+warps-1)//warps,),(warps,),"TTTTTiff")
    L2(w_m, i_m, s_m, logits, bias, T, True, 1.5)
    torch.cuda.synchronize()
    dw = float((w_ref-w_m).abs().max())
    di = int((i_ref!=i_m).sum())
    u_ref = bench(lambda: torch.ops._moe_C.topk_softplus_sqrt(w_ref,i_ref,tei,logits,True,1.5,bias,None,None,None,None,0))
    u_m = bench(lambda: L2(w_m, i_m, s_m, logits, bias, T, True, 1.5))
    print(f"T={T:<4} wdiff={dw:.6f} idx mismatch={di}/{T*K}   _C={u_ref:6.2f}us ours={u_m:6.2f}us ({u_ref/u_m:.2f}x)")
