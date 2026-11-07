import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.deepseek.moe_fused import fp4_swiglu_quant_cuda, _packed
from tinyinfer.deepseek import fp4_native
from tinyinfer.kernels.quant import act_quant
torch.manual_seed(30)
N, K = 256, 512
x = torch.randn(1, K, dtype=torch.bfloat16, device='cuda')
aq, asc = fp4_native.quantize_act(x, K)
w1 = torch.randint(0, 256, (N, K//2), dtype=torch.uint8, device='cuda')
w3 = torch.randint(0, 256, (N, K//2), dtype=torch.uint8, device='cuda')
s1 = torch.randint(120, 132, (N, K//32), dtype=torch.uint8, device='cuda')
s3 = torch.randint(120, 132, (N, K//32), dtype=torch.uint8, device='cuda')
limit = 4.0
hq, hs = fp4_swiglu_quant_cuda(aq, asc, w1, s1, w3, s3, limit)
print(f'RESULT hq dequant[:6] (uses packed layout):')
e8 = lambda b: torch.pow(2.0, (b.view(torch.uint8).to(torch.int32)-127).float())
s = e8(hs).repeat_interleave(32)[:N]
print('  ', (hq.float()*s)[:6].tolist())
# 用 packed 后的权重做参考
print('RESULT packed shape same?', _packed(w1).shape == w1.shape)