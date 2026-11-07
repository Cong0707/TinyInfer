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
t0 = bench(lambda: M.MarlinFp4Weight(w, s))
print('RESULT MarlinFp4Weight total: %.1f us' % (t0*1e6))
packed = w.reshape(N, K//2).contiguous()
t1 = bench(lambda: packed.view(torch.int32).T.contiguous())
print('RESULT   transpose          : %.1f us' % (t1*1e6))
e8 = s.reshape(N, K//32).view(torch.uint8).view(torch.float8_e8m0fnu)
t2 = bench(lambda: M._prep_scales(e8, K, N))
print('RESULT   prep_scales        : %.1f us' % (t2*1e6))
got = M._repack_launcher()
if got:
    launch, sms = got
    src = packed.view(torch.int32).T.contiguous()
    out = torch.empty(K//16, N*16//8, dtype=torch.int32, device='cuda')
    t3 = bench(lambda: launch(src, out, K, N))
    print('RESULT   repack kernel      : %.1f us' % (t3*1e6))
print('RESULT   剩余（alloc/dispatch）= %.1f us' % ((t0-t1-t2-(t3 if got else 0))*1e6))