import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
import tinyinfer.deepseek.gemv_fp4_marlin as M
N, K = 2304, 5120
k32 = K//32
# 用递增序列当输入，看输出怎么排
s = torch.arange(N*k32, dtype=torch.int32, device='cuda').to(torch.int32).view(N,k32).view(torch.int32)
import numpy as np
e8 = s.view(torch.float8_e8m0fnu) if False else s
# 直接调 _permute_scales (它只做 reshape/permute，不改值)
t1 = s.T.contiguous()
t2 = M._permute_scales(t1, K, N, 32)
print(f'RESULT t1[0,:6] = {t1[0,:6].tolist()}')
print(f'RESULT t2[0,:6] = {t2[0,:6].tolist()}')
print(f'RESULT t1[0,:6] 意味着从 s 的 [(0..5), 0] 取')
print(f'RESULT 应为 s[0,0], s[1,0] ... = {s[:6,0].tolist()}')
print()
print(f'RESULT t2 是 8x8 block permute 后：')
print(f'  t2[0,:6] = {t2[0,:6].tolist()}')
print(f'  对比 s[:,0] 前 16 个 = {s[:16,0].tolist()}')