import sys, torch, time
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
import tinyinfer.deepseek.gemv_fp4_marlin as M
N, K = 2304, 5120
w = torch.randint(0,256,(N,K//2),dtype=torch.uint8,device='cuda')
s = torch.randint(120,132,(N,K//32),dtype=torch.uint8,device='cuda')
def bench(fn, n=30):
    for _ in range(3): fn()
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t0)/n
packed = w.reshape(N, K//2).contiguous()
t1 = bench(lambda: packed.view(torch.int32).T.contiguous())
print('RESULT .T.contiguous()  : %7.1f us' % (t1*1e6))
e8 = s.reshape(N, K//32)
t2 = bench(lambda: e8.T.contiguous())
print('RESULT scales .T        : %7.1f us' % (t2*1e6))
t3 = bench(lambda: M._permute_scales(e8.T.contiguous(), K, N, 32))
print('RESULT _permute_scales  : %7.1f us' % (t3*1e6))
t4 = bench(lambda: M._mxfp4_scales(s))
print('RESULT _mxfp4_scales    : %7.1f us' % (t4*1e6))
t5 = bench(lambda: torch.empty(K//16, N*16//8, dtype=torch.int32, device='cuda'))
print('RESULT empty out        : %7.1f us' % (t5*1e6))
tot = (t1+t2+t3+t4+t5)*1e6
print('RESULT sum of parts     : %7.1f us  (实测 _prepared ~187)' % tot)