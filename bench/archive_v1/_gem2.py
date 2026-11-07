import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.weights import ShardStore
from tinyinfer import gemm
from tinyinfer.kernels import native
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
store = ShardStore(d)
w = store.get('layers.3.ffn.shared_experts.w1.weight').cuda()
ws = store.get('layers.3.ffn.shared_experts.w1.scale').cuda()
N, K = w.shape
x = torch.randn(1, K, dtype=torch.bfloat16, device='cuda')
y1 = gemm.fp8_linear(x, w, ws)
# 慢路径复现
from tinyinfer.kernels.quant import act_quant
aq, asc = act_quant(x, 32, 'ue8m0', torch.float8_e8m0fnu)
y2 = torch.empty(N, dtype=torch.bfloat16, device='cuda')
native.launch(native.kernel('fp8_gemv_bs','fp8_gemv_bs'), ((N+15)//16,), (512,),
              [aq.view(torch.uint8).reshape(-1), asc.view(torch.uint8).reshape(-1),
               w.view(torch.uint8).reshape(-1), ws.view(torch.uint8).reshape(-1), y2, N, K])
torch.cuda.synchronize()
print(f'RESULT identical: {torch.equal(y1.view(-1), y2)}')