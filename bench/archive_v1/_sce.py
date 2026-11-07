import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
import tinyinfer.deepseek.gemv_fp4_marlin as M
N, K = 2304, 5120
k32 = K//32
ids = torch.arange(N*k32, dtype=torch.int32, device='cuda').view(N, k32)
t1 = ids.T.contiguous()
t2 = M._permute_scales(t1, K, N, 32)
print('RESULT t1[0][:8] =', (t1[0][:8]//k32).tolist())
print('RESULT t2[0][:8] =', (t2[0][:8]//k32).tolist())
print()
print('RESULT t2[0][0..63] 的 k_src =', (t2[0][:64]%k32).tolist())
print('RESULT t2[1][0..15] 的 k_src =', (t2[1][:16]%k32).tolist())
print()
print('RESULT => _permute_scales 对 k 维做了什么？')
print('  t1 的 k 是行(160)，t2 也是 160 行')
print('  若 perm 生效，t2[0] 应取 t1[perm64(0)]=t1[0]')
print('  t2[1] 应取 t1[perm64(1)]=t1[8]')
print('  验证 t2[1][0] =', (t2[1][0]//k32).item(), ' t1[8][0] =', (t1[8][0]//k32).item())