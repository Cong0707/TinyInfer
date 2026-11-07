import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.deepseek.moe_fused import fp4_swiglu_quant_cuda
from tinyinfer.deepseek import fp4_native
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
g = fp4_native.gemv_fp4_pq(aq[0], asc[0], w1, s1)
u = fp4_native.gemv_fp4_pq(aq[0], asc[0], w3, s3)
print('RESULT g[:4]', g.flatten()[:4].tolist())
print('RESULT u[:4]', u.flatten()[:4].tolist())
print('RESULT hq[:6]', hq[:6].tolist(), 'scale', hs[:2].tolist())
# 反量化 hq 看实际值
e8 = lambda b: torch.pow(2.0, (b.view(torch.uint8).to(torch.int32)-127).float())
s = e8(hs).repeat_interleave(32)[:N]
dec = hq.float() * s
print('RESULT hq dequant[:6]', dec[:6].tolist())
# 参考 h
g_ = g.flatten().float().clamp(max=limit)
u_ = u.flatten().float().clamp(-limit, limit)
hr = (g_/(1+torch.exp(-g_)))*u_
print('RESULT ref h[:6]', hr[:6].tolist())
print('RESULT diff', (dec - hr).abs().max().item())