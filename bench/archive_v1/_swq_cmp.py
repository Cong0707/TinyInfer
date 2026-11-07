
import torch
from tinyinfer.deepseek.moe_fused import fused_fp4_swiglu_pq, fp4_swiglu_quant_cuda
from tinyinfer.official_kernel_ref import act_quant

torch.manual_seed(0)
K, N = 7168, 2048
nb = K // 32
dev = "cuda"

# activation: quantise a random bf16 row the official way
x = torch.randn(1, K, dtype=torch.bfloat16, device=dev) * 3
aq, asc = act_quant(x, 32, "ue8m0", torch.float8_e8m0fnu)

w1 = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=dev)
w3 = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=dev)
s1 = torch.full((N, nb), 127, dtype=torch.uint8, device=dev).view(torch.float8_e8m0fnu)
s3 = torch.full((N, nb), 127, dtype=torch.uint8, device=dev).view(torch.float8_e8m0fnu)

fp4 = torch.float4_e2m1fn_x2
h = fused_fp4_swiglu_pq(aq, asc, w1.view(fp4), s1, w3.view(fp4), s3, 10.0)
hq, hs = act_quant(h, 32, "ue8m0", torch.float8_e8m0fnu)
torch.cuda.synchronize()
print("ref  hq:", tuple(hq.shape), hq.dtype, " hs:", tuple(hs.shape))

got = fp4_swiglu_quant_cuda(aq, asc, w1.view(fp4), s1, w3.view(fp4), s3, 10.0)
if got is None:
    print("CUDA path returned None (fell back)")
else:
    hq2, hs2 = got
    torch.cuda.synchronize()
    print("ours hq:", tuple(hq2.shape), hq2.dtype, " hs:", tuple(hs2.shape))
    a = hq.float(); b = hq2.float()
    print(f"hq maxdiff = {(a-b).abs().max().item():.3e}   bit={bool((hq.view(torch.uint8)==hq2.view(torch.uint8)).all())}")
    s_a = hs.view(torch.uint8).to(torch.int32); s_b = hs2.view(torch.uint8).to(torch.int32)
    print(f"hs maxdiff = {(s_a-s_b).abs().max().item()}  (scale exponent bytes)")
