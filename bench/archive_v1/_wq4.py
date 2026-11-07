import sys, torch, time
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.weights import ShardStore
from tinyinfer.kernels import native
from tinyinfer.gemm import SCALE_FMT, SCALE_DTYPE, _act_quant, _bf16_default
import kernel as _k
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
st = ShardStore(d)
w = st.get('layers.3.attn.wq_b.weight').to('cuda'); s = st.get('layers.3.attn.wq_b.scale').to('cuda')
N, K = w.shape
x = torch.randn(1, K, dtype=torch.bfloat16, device='cuda')
with _bf16_default():
    aq, asc = _act_quant(x, 32, SCALE_FMT, SCALE_DTYPE)
y = torch.empty(N, device='cuda', dtype=torch.bfloat16)
fn = native.kernel('fp8_gemv_bs','fp8_gemv_bs')
args = [aq.view(torch.uint8).reshape(-1), asc.view(torch.uint8).reshape(-1),
        w.view(torch.uint8).reshape(-1), s.view(torch.uint8).reshape(-1), y, N, K]
for _ in range(3): native.launch(fn, ((N+15)//16,), (512,), args)
torch.cuda.synchronize(); t0=time.perf_counter()
for _ in range(20): native.launch(fn, ((N+15)//16,), (512,), args)
torch.cuda.synchronize(); t1=(time.perf_counter()-t0)/20*1e6
with _bf16_default():
    for _ in range(3): _k.fp8_gemm(aq, asc, w, s, SCALE_DTYPE, block_size=32)
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(20): _k.fp8_gemm(aq, asc, w, s, SCALE_DTYPE, block_size=32)
    torch.cuda.synchronize(); t2=(time.perf_counter()-t0)/20*1e6
print(f'RESULT native={t1:.0f}us  tilelang={t2:.0f}us  ratio={t2/t1:.2f}x')