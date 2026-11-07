import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
import tinyinfer.deepseek.gemv_fp4_marlin as M
N, K = 2304, 5120
k32 = K//32
ids = torch.arange(N*k32, dtype=torch.int32, device='cuda').view(N, k32)
t3 = M._permute_scales(ids.T.contiguous(), K, N, 32)
t3 = t3.view(-1,4)[:, [0,2,1,3]].view(t3.size(0), -1) if False else t3
# 注意上面没做 perm4，单独做
t2 = M._permute_scales(ids.T.contiguous(), K, N, 32)
t3 = t2.view(-1,4)[:, [0,2,1,3]].view(t2.size(0), -1)
ns = (t3[0]//k32).tolist()
# 找公式
def guess(n):
    c = n // 64; r = n % 64
    a = r % 8; b = r // 8
    return c*64 + (a%2)*32 + (a//2)*8 + (b%2)*16 + (b//2)*4 + ((b//2)%2)*2 + 0
ok = all(guess(n) == ns[n] for n in range(64))
print(f'RESULT guess formula matches: {ok}')
if not ok:
    for n in range(8): print(f'  n={n}: want={ns[n]} guess={guess(n)}')
# 直接看位模式
print(f'RESULT n=0..7 -> {ns[0:8]}')
print(f'RESULT 二进制：')
for n in range(8):
    print(f'  n={n:03b} -> {ns[n]:06b} = {ns[n]}')