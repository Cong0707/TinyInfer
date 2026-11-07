import sys, torch, time
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.deepseek.gemv_fp4_marlin import gemv_fp4_fast, _prepared
N, K = 2304, 5120
w = torch.randint(0,256,(N,K//2),dtype=torch.uint8,device='cuda')
s = torch.randint(120,132,(N,K//32),dtype=torch.uint8,device='cuda')
x = torch.randn(1,K,dtype=torch.bfloat16,device='cuda')
def bench(fn, n=30):
    for _ in range(3): fn()
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t0)/n
# 1. 重复同一个 tensor -> 应该命中 cache
t1 = bench(lambda: gemv_fp4_fast(x, w, s))
print(f'RESULT same w x30 : {t1*1e6:6.1f} us  (cache hit)')
# 2. 交替两个不同 tensor -> cache miss 每次重 repack
w2 = torch.randint(0,256,(N,K//2),dtype=torch.uint8,device='cuda')
s2 = torch.randint(120,132,(N,K//32),dtype=torch.uint8,device='cuda')
def alt():
    gemv_fp4_fast(x, w, s)
    gemv_fp4_fast(x, w2, s2)
t2 = bench(alt, 20)
print(f'RESULT alternating: {t2*1e6/2:6.1f} us/call (cache miss + repack)')
print(f'RESULT repack cost : {(t2/2 - t1)*1e6:.0f} us')