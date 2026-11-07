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
print('RESULT n_src 表 (out col -> src col):')
for i in range(0, 16):
    print(f'  out[{i}] <- src col {ns[i]}, row {ks[i]}')
print()
print('RESULT 用公式：(out col i) 的 src col = ?')
print(f'  i=0->{ns[0]} i=1->{ns[1]} i=2->{ns[2]} i=3->{ns[3]}')
print(f'  i=8->{ns[8]} i=9->{ns[9]} i=16->{ns[16]} i=17->{ns[17]}')
print()
print('RESULT 观察：')
print('  i=0..7 -> 0,16,8,24,32,48,40,56 = ((i&1)*16 + ((i>>1)&1)*8 + ((i>>2)&1)*32)')
print('  即 bit0->bit4, bit1->bit3, bit2->bit5')
print('  i=8..15 -> 1,17,9,25,33,49,41,57 = 上面 +1')
print('  => src_col = (i//8) + bits_rearranged(i%8)')