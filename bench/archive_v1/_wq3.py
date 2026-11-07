import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.weights import ShardStore
from tinyinfer.kernels import native
from tinyinfer.gemm import SCALE_FMT, SCALE_DTYPE, _act_quant, _bf16_default
import kernel as _k
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
st = ShardStore(d)
w = st.get('layers.3.attn.wq_b.weight').to('cuda'); s = st.get('layers.3.attn.wq_b.scale').to('cuda')
N, K = w.shape
torch.manual_seed(0)
x = torch.randn(1, K, dtype=torch.bfloat16, device='cuda')
with _bf16_default():
    aq, asc = _act_quant(x, 32, SCALE_FMT, SCALE_DTYPE)
    ref = _k.fp8_gemm(aq, asc, w, s, SCALE_DTYPE, block_size=32)
y = torch.empty(N, device='cuda', dtype=torch.bfloat16)
native.launch(native.kernel('fp8_gemv_bs','fp8_gemv_bs'), ((N+15)//16,), (512,),
              [aq.view(torch.uint8).reshape(-1), asc.view(torch.uint8).reshape(-1),
               w.view(torch.uint8).reshape(-1), s.view(torch.uint8).reshape(-1), y, N, K])
torch.cuda.synchronize()
a = ref.flatten().float(); b = y.float()
d_ = (a != b)
print(f'RESULT diff_elems={int(d_.sum())}/{N} ({100*d_.float().mean().item():.3f}%) maxdiff={(a-b).abs().max().item():.3e}')
idx = d_.nonzero().flatten()
print('first idx:', idx[:8].tolist())
print('row mod 32:', sorted(set((i.item()%32) for i in idx[:200])))
print('row mod 16:', sorted(set((i.item()%16) for i in idx[:200])))