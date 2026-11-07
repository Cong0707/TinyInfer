import sys, torch, time
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.weights import ShardStore
from tinyinfer.kernels import native
from tinyinfer.gemm import SCALE_FMT, SCALE_DTYPE, _act_quant, _bf16_default
import kernel as _k
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
st = ShardStore(d)
def bench(fn, n=8):
    for _ in range(3): fn()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t0) / n * 1e6
for name in ('layers.3.attn.wq_b', 'layers.5.ffn.shared_experts.w1'):
    w = st.get(name + '.weight').to('cuda'); s = st.get(name + '.scale').to('cuda')
    N, K = w.shape
    print(f'=== {name.split(".",2)[-1]}  N={N} K={K}', flush=True)
    for M in (1, 2, 4, 8, 16, 32, 64):
        x = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
        with _bf16_default():
            aq, asc = _act_quant(x, 32, SCALE_FMT, SCALE_DTYPE)
            tc = bench(lambda: _k.fp8_gemm(aq, asc, w, s, SCALE_DTYPE, block_size=32))
        line = f'  M={M:3d}  tensorcore={tc:8.1f}us'
        if M == 1:
            y = torch.empty(N, device='cuda', dtype=torch.bfloat16)
            fn = native.kernel('fp8_gemv_bs','fp8_gemv_bs')
            args = [aq.view(torch.uint8).reshape(-1), asc.view(torch.uint8).reshape(-1),
                    w.view(torch.uint8).reshape(-1), s.view(torch.uint8).reshape(-1), y, N, K]
            sm = bench(lambda: native.launch(fn, ((N+15)//16,), (512,), args))
            line += f'   simt={sm:8.1f}us   simt/tc={sm/tc:.2f}x'
        print(line, flush=True)