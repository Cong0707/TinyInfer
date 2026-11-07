import sys, torch, time
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.kernels import native, mma_gemm
from tinyinfer.weights import ShardStore
from tinyinfer.gemm import SCALE_FMT, SCALE_DTYPE, _act_quant
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
st = ShardStore(d)
def bench(fn, n=20):
    for _ in range(3): fn()
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t0)/n*1e6
for name in ('layers.3.attn.wq_b', 'layers.5.ffn.shared_experts.w1'):
    w = st.get(name + '.weight').to('cuda'); s = st.get(name + '.scale').to('cuda')
    N, K = w.shape
    if N % 64: print(name, 'skip N%64'); continue
    x = torch.randn(1, K, dtype=torch.bfloat16, device='cuda')
    aq, asc = _act_quant(x, 32, SCALE_FMT, SCALE_DTYPE)
    y = torch.empty(N, device='cuda', dtype=torch.bfloat16)
    fnk = native.kernel('fp8_gemv_bs','fp8_gemv_bs')
    args = [aq.view(torch.uint8).reshape(-1), asc.view(torch.uint8).reshape(-1),
            w.view(torch.uint8).reshape(-1), s.view(torch.uint8).reshape(-1), y, N, K]
    t1 = bench(lambda: native.launch(fnk, ((N+15)//16,), (512,), args))
    t2 = bench(lambda: mma_gemm.fp8_gemm_mma(x, w, s))
    print(f'RESULT {name.split(".",2)[-1]:22s} N={N:5d} K={K:4d} simt={t1:7.1f}us mma={t2:7.1f}us ratio={t2/t1:.2f}x')