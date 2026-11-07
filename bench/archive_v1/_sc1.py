import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
import tinyinfer.deepseek.gemv_fp4_marlin as M
N, K = 2304, 5120
s = torch.randint(120,132,(N,K//32),dtype=torch.uint8,device='cuda')
e8 = s.reshape(N, K//32).view(torch.float8_e8m0fnu)
# 参考：三步
ref = M._mxfp4_scales(M._permute_scales(e8.T.contiguous(), K, N, 32))
print(f'RESULT ref shape={tuple(ref.shape)} dtype={ref.dtype}')
print(f'RESULT ref[:8]={ref.view(torch.uint8).flatten()[:8].tolist()}')
# 只看 transpose 之后
t = e8.T.contiguous()
print(f'RESULT after .T: {tuple(t.shape)} {t.view(torch.uint8)[0,:6].tolist()}')
print(f'RESULT orig[0,:6]: {s[0,:6].tolist()}')
print(f'RESULT orig[:6,0]: {s[:6,0].tolist()}')