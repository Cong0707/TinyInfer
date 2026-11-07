import sys, torch, numpy as np
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
import tinyinfer.deepseek.gemv_fp4_marlin as M
N, K = 2304, 5120
k32 = K//32
ids = torch.arange(N*k32, dtype=torch.int32, device='cuda').view(N, k32)
t1 = ids.T.contiguous()
t2 = M._permute_scales(t1, K, N, 32)
t3 = t2.view(-1,4)[:, [0,2,1,3]].view(t2.size(0), -1)
# n_src 表：(t3[0][n] // k32)
ns = (t3[0]//k32).tolist()
ks = (t3[0]%k32).tolist()
print(f'RESULT n_src[0:16] = {ns[:16]}')
print(f'RESULT n_src[16:32] = {ns[16:32]}')
print(f'RESULT n_src[32:48] = {ns[32:48]}')
print(f'RESULT n_src[48:64] = {ns[48:64]}')
print(f'RESULT n_src[64:72] = {ns[64:72]}')
print()
print(f'RESULT 验证 perm4 假设: t3[n] 应 = t2[perm4(n)]')
print(f'  t2[0][0:8] = {(t2[0][:8]//k32).tolist()}')
print(f'  t3[0][0:8] = {ns[:8]}')
print(f'  => t3[n]=t2[m], m = {[0,2,1,3,4,6,5,7]} ?')
print(f'  t2[[0,2,1,3,4,6,5,7]] = {(t2[0][[0,2,1,3,4,6,5,7]]//k32).tolist()}')