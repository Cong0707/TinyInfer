import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
import tinyinfer.deepseek.gemv_fp4_marlin as M
N, K = 2304, 5120
k32 = K//32
ids = torch.arange(N*k32, dtype=torch.int32, device='cuda').view(N, k32)
t2 = M._permute_scales(ids.T.contiguous(), K, N, 32)
t3 = t2.view(-1,4)[:, [0,2,1,3]].view(t2.size(0), -1)
print('RESULT out[k][0] <- src[n][k_src]：')
for k in range(16):
    raw = t3[k][0].item()
    print(f'  out[k={k:2d}][0] <- src[{raw//k32}][{raw%k32}]')
print()
print('RESULT n 维也应该随 k 变？检查 out[k][0] 的 n_src：')
for k in range(4):
    print(f'  k={k}: n_src={(t3[k][:4]//k32).tolist()}')