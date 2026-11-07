import sys, torch, time
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.kernels import native, mma_gemm
from tinyinfer.weights import ShardStore
from tinyinfer.gemm import SCALE_FMT, SCALE_DTYPE, _act_quant, _bf16_default
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
st = ShardStore(d)
name = 'layers.3.attn.wq_b'
w = st.get(name + '.weight').to('cuda'); s = st.get(name + '.scale').to('cuda')
N, K = w.shape
print('N', N, 'K', K, 'N%64', N%64)
x = torch.randn(1, K, dtype=torch.bfloat16, device='cuda')
aq, asc = _act_quant(x, 32, SCALE_FMT, SCALE_DTYPE)
y1 = torch.empty(N, device='cuda', dtype=torch.bfloat16)
fn = native.kernel('fp8_gemv_bs','fp8_gemv_bs')
native.launch(fn, ((N+15)//16,), (512,), [aq.view(torch.uint8).reshape(-1), asc.view(torch.uint8).reshape(-1),
               w.view(torch.uint8).reshape(-1), s.view(torch.uint8).reshape(-1), y1, N, K])
try:
    y2 = mma_gemm.fp8_gemm_mma(x, w, s)
    torch.cuda.synchronize()
    a = y1.float(); b = y2.flatten().float()
    print(f'RESULT M=1 simt-vs-mma diff={(a!=b).sum().item()}/{N} max={(a-b).abs().max().item():.2e}')
except Exception as e:
    print('mma failed:', repr(e)[:120])