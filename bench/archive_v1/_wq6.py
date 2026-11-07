import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.weights import ShardStore
from tinyinfer.kernels import native
from tinyinfer.gemm import SCALE_FMT, SCALE_DTYPE, _act_quant, _bf16_default
import kernel as _k
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
st = ShardStore(d)
torch.manual_seed(0)
for name in ('layers.3.attn.wq_b', 'layers.5.ffn.shared_experts.w2'):
    w = st.get(name + '.weight').to('cuda'); s = st.get(name + '.scale').to('cuda')
    N, K = w.shape
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
    print(f'RESULT {name:44s} K={K:4d} diff={(a!=b).sum().item()}/{N} max={(a-b).abs().max().item():.2e}')