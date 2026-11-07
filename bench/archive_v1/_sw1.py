import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.deepseek.moe_fused import fp4_swiglu_quant_cuda
from tinyinfer.deepseek import fp4_native
import torch.nn.functional as F
torch.manual_seed(30)
N, K = 256, 512
# 激活：fp8 量化 + e8m0 scale
x = torch.randn(1, K, dtype=torch.bfloat16, device='cuda')
aq, asc = fp4_native.quantize_act(x, K)
w1 = torch.randint(0, 256, (N, K//2), dtype=torch.uint8, device='cuda')
w3 = torch.randint(0, 256, (N, K//2), dtype=torch.uint8, device='cuda')
s1 = torch.randint(120, 132, (N, K//32), dtype=torch.uint8, device='cuda')
s3 = torch.randint(120, 132, (N, K//32), dtype=torch.uint8, device='cuda')
limit = 4.0
try:
    hq, hs = fp4_swiglu_quant_cuda(aq, asc, w1, s1, w3, s3, limit)
    print(f'RESULT out q={tuple(hq.shape)} dtype={hq.dtype} s={tuple(hs.shape)}')
except Exception as e:
    import traceback; traceback.print_exc()