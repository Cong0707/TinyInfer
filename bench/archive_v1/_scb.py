import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
import tinyinfer.deepseek.gemv_fp4_marlin as M
N, K = 2304, 5120
k32 = K//32
ids = torch.arange(N*k32, dtype=torch.int32, device='cuda').view(N, k32)
t2 = M._permute_scales(ids.T.contiguous(), K, N, 32)
t3 = t2.view(-1,4)[:, [0,2,1,3]].view(t2.size(0), -1)
ns = (t3[0]//k32).tolist()
ks = (t3[0]%k32).tolist()
def n_src(n):
    a = n & 7; c = n >> 6
    # bit0<->bit1 swap within a, then <<3, plus the c part
    swapped = (a & 4) | ((a & 1) << 1) | ((a & 2) >> 1)
    return c*64 + swapped*8
ok = all(n_src(n) == ns[n] for n in range(128))
print(f'RESULT n_src formula over 128: {ok}')
if not ok:
    bad = [n for n in range(128) if n_src(n)!=ns[n]][:5]
    for n in bad: print(f'  n={n} want={ns[n]} got={n_src(n)}')
def k_src(k):
    return (k//64)*64 + (k%8)*8 + ((k%64)//8)
ok2 = all(k_src(k) == ks[k] for k in range(160))
print(f'RESULT k_src formula over 160: {ok2}')
if not ok2:
    bad = [k for k in range(160) if k_src(k)!=ks[k]][:5]
    for k in bad: print(f'  k={k} want={ks[k]} got={k_src(k)}')