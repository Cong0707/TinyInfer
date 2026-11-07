
import torch, sys
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.deepseek.moe_fused import fp4_swiglu_quant_cuda
from tinyinfer.deepseek.gemv_fp4 import quantize_act

# 合成数据，不加载模型
torch.manual_seed(0)
N, K = 2048, 512
aq1, asc1 = quantize_act(torch.randn(1, K, dtype=torch.bfloat16, device="cuda"), K)
aq2 = aq1.repeat(2, 1).contiguous(); asc2 = asc1.repeat(2, 1).contiguous()

w1 = torch.randint(0, 256, (N, K//2), dtype=torch.uint8, device="cuda")
s1 = torch.full((N, K//32), 127, dtype=torch.uint8, device="cuda")
W = w1.view(torch.float4_e2m1fn_x2); S = s1.view(torch.float8_e8m0fnu)

from tinyinfer.deepseek.gemv_fp4 import gemv_fp4_pq
y1 = gemv_fp4_pq(aq1[0], asc1[0], W, S)
rows = [gemv_fp4_pq(aq2[i], asc2[i], W, S) for i in range(2)]
y2 = torch.cat(rows, 0)
print("逐行 M=2 vs 单行 M=1 第0行 maxdiff:", (y2[0].float()-y1.float()).abs().max().item())
print("两行是否相同:", (y2[0]==y2[1]).all().item())
