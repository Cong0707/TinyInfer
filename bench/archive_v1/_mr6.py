import sys, torch, time
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
import tinyinfer.deepseek.gemv_fp4_marlin as M
N, K = 2304, 5120
x = torch.randn(1,K,dtype=torch.bfloat16,device='cuda')
ws = []
for i in range(6):
    w = torch.randint(0,256,(N,K//2),dtype=torch.uint8,device='cuda')
    s = torch.randint(120,132,(N,K//32),dtype=torch.uint8,device='cuda')
    ws.append((w,s))
def bench(fn, n=10):
    for _ in range(2): fn()
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t0)/n
def cur():
    for w,s in ws: M.gemv_fp4_fast(x, w, s)
t1 = bench(cur)
print('RESULT cache_max=1, 6 experts: %.2f ms/layer' % (t1*1000))
prep = [M.MarlinFp4Weight(w,s) for w,s in ws]
launch, smem = M._native_launcher()
wsp = M._workspace(x.device)
def cached():
    for p in prep:
        out = torch.empty(1, N, dtype=torch.bfloat16, device='cuda')
        launch(x, p.qweight, out, None, None, None, p.scales, None, None,
               1, N, K, x.stride(0), wsp, 0,0,0, smem)
t2 = bench(cached)
print('RESULT 6 precomputed  : %.2f ms/layer' % (t2*1000))
print('RESULT speedup        : %.1fx' % (t1/t2))