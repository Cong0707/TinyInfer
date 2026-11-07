import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.deepseek.fp4_native import gemv_fp4_pq, quantize_act
N, K = 2304, 5120
torch.manual_seed(0)
w = torch.randint(0,256,(N,K//2),dtype=torch.uint8,device='cuda')
s = torch.randint(120,132,(N,K//32),dtype=torch.uint8,device='cuda')
x = torch.randn(1,K,dtype=torch.bfloat16,device='cuda')
aq, asc = quantize_act(x, K)
print(f'RESULT aq dtype={aq.dtype} shape={tuple(aq.shape)}')
print(f'RESULT asc dtype={asc.dtype} shape={tuple(asc.shape)}')
print(f'RESULT asc[:4]={asc.view(torch.uint8).flatten()[:4].tolist() if asc.dtype==torch.uint8 else asc.flatten()[:4].tolist()}')
out = gemv_fp4_pq(aq[0], asc[0], w, s)
print(f'RESULT out[:3]={out[0,:3].tolist()}')
# 检查 aq 范围
af = aq.view(torch.float8_e4m3fn).float()
print(f'RESULT aq float range [{af.min().item():.2f}, {af.max().item():.2f}]')
print(f'RESULT x range [{x.min().item():.2f}, {x.max().item():.2f}]')