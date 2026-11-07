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
# repack kernel 本身
got = M._repack_launcher()
print('RESULT repack launcher:', 'native' if got else 'fallback pyd')
t_alloc = bench(lambda: torch.empty(K//16, N*16//8, dtype=torch.int32, device='cuda'))
print(f'RESULT empty alloc   : {t_alloc*1e6:7.1f} us')
if got:
    launch, sms = got
    out = torch.empty(K//16, N*16//8, dtype=torch.int32, device='cuda')
    t_k = bench(lambda: launch(w, out, K, N))
    print(f'RESULT repack kernel : {t_k*1e6:7.1f} us')
t_s = bench(lambda: M._mxfp4_scales(s))
print(f'RESULT scales permute: {t_s*1e6:7.1f} us')
t_p = bench(lambda: M._prepared(w, s, None))
print(f'RESULT _prepared     : {t_p*1e6:7.1f} us')