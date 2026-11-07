
import torch
from tinyinfer.deepseek.moe_fused import fused_fp4_swiglu_pq, fp4_swiglu_quant_cuda
from tinyinfer.official_kernel_ref import act_quant

torch.manual_seed(0)
K, N = 7168, 2048
nb = K // 32
x = torch.randn(1, K, dtype=torch.bfloat16, device="cuda") * 3
aq, asc = act_quant(x, 32, "ue8m0", torch.float8_e8m0fnu)
w1 = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device="cuda")
w3 = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device="cuda")
s1 = torch.full((N, nb), 127, dtype=torch.uint8, device="cuda").view(torch.float8_e8m0fnu)
s3 = torch.full((N, nb), 127, dtype=torch.uint8, device="cuda").view(torch.float8_e8m0fnu)
fp4 = torch.float4_e2m1fn_x2

h = fused_fp4_swiglu_pq(aq, asc, w1.view(fp4), s1, w3.view(fp4), s3, 10.0)
hq, hs = act_quant(h, 32, "ue8m0", torch.float8_e8m0fnu)
hq2, hs2 = fp4_swiglu_quant_cuda(aq, asc, w1.view(fp4), s1, w3.view(fp4), s3, 10.0)
torch.cuda.synchronize()

scale = hs[0].float()                      # e8m0 -> float
print("h  ref[:6] :", [round(v,4) for v in h[0,:6].float().tolist()])
print("hq ref[:6] :", [round(v,4) for v in hq[0,:6].float().tolist()])
print("hq our[:6] :", [round(v,4) for v in hq2[:6].float().tolist()])
print("h*inv ref  :", [round(v/scale[0].item(),4) for v in h[0,:6].float().tolist()])
print("scale ref[0]:", round(scale[0].item(),6), " our[0]:", round(hs2[0].float().item(),6))
print("h amax =", round(h[0].float().abs().max().item(),4))
