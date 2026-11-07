import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.deepseek.moe_fused import fp4_swiglu_quant_cuda
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
g = fp4_native.gemv_fp4_pq(aq[0], asc[0], w1, s1)
u = fp4_native.gemv_fp4_pq(aq[0], asc[0], w3, s3)
# 逐位复刻 kernel：bf16 round-trip, clamp, g/(1+exp(-g))*u, bf16
g_ = g.float().to(torch.bfloat16).float()
u_ = u.float().to(torch.bfloat16).float()
g_ = g_.clamp(max=limit)
u_ = u_.clamp(-limit, limit)
v = (g_ / (1 + torch.exp(-g_))) * u_
h = v.to(torch.bfloat16)
rq, rs = act_quant(h.reshape(1, N), 32, 'ue8m0', torch.float8_e8m0fnu)
dq = (hq.view(torch.uint8) != rq.reshape(-1).view(torch.uint8)).sum().item()
e8 = lambda b: torch.pow(2.0, (b.view(torch.uint8).to(torch.int32)-127).float())
ds = (e8(hs) != e8(rs.reshape(-1))).sum().item()
print(f'RESULT fp8 diff={dq}/{hq.numel()}  scale diff={ds}/{hs.numel()}')