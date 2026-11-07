import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
import tinyinfer.deepseek.gemv_fp4_marlin as M
N, K = 2304, 5120
k32 = K//32
# 用 int32 递增序列（避免 e8m0 类型问题）
ids = torch.arange(N*k32, dtype=torch.int32, device='cuda').view(N, k32)
t1 = ids.T.contiguous()
t2 = M._permute_scales(t1, K, N, 32)
t3 = t2.view(-1,4)[:, [0,2,1,3]].view(t2.size(0), -1)
print(f'RESULT t3 shape {tuple(t3.shape)}')
print(f'RESULT t3[0,:8] = {(t3[0,:8]//k32).tolist()}  (n index)')
print(f'RESULT t3[0,:8] % k32 = {(t3[0,:8]%k32).tolist()}  (k index)')
print(f'RESULT t3[0,:8] raw = {t3[0,:8].tolist()}')
# 输出 out[k][n] = ?  由 t3[k][n] = ids[n_src][k_src] = n_src*k32 + k_src
out = torch.empty(k32, N, dtype=torch.int32, device='cuda')
first = t3[0].tolist()
print(f'RESULT out[0][0..7] 应来自 raw {first[:8]}')
n_s = [v//k32 for v in first[:8]]
k_s = [v%k32 for v in first[:8]]
print(f'RESULT n_src = {n_s}')
print(f'RESULT k_src = {k_s}')