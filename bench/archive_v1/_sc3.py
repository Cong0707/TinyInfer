import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
import tinyinfer.deepseek.gemv_fp4_marlin as M
N, K = 2304, 5120
n32, k32 = N//32, K//32
print(f'RESULT n32={n32} k32={k32} total={n32*k32}')
s = torch.randint(120,132,(N,K//32),dtype=torch.uint8,device='cuda')
e8 = s.reshape(N, k32).view(torch.float8_e8m0fnu)
t1 = e8.T.contiguous()
print(f'RESULT .T shape: {tuple(t1.shape)} numel={t1.numel()}')
t2 = M._permute_scales(t1, K, N, 32)
print(f'RESULT _permute shape: {tuple(t2.shape)} numel={t2.numel()}')
t3 = M._mxfp4_scales(t2)
print(f'RESULT _mxfp4 shape: {tuple(t3.shape)} numel={t3.numel()}')
print(f'RESULT 所以 kernel 输出应该是 {tuple(t3.shape)}')