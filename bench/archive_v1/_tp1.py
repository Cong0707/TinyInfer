import sys, torch, time
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.kernels import native
N, K = 2304, 5120
K2 = K//2
w = torch.randint(0,256,(N,K2),dtype=torch.uint8,device='cuda')
# 参考
ref = w.view(torch.int32).T.contiguous()
print(f'RESULT ref shape {tuple(ref.shape)}')
# 我的 kernel
K8 = K//8
out = torch.empty(K8, N, dtype=torch.int32, device='cuda')
grid = ((K2+31)//32, (N+31)//32)
b = native.bind('transpose_4bit','transpose_4bit', grid, (32,32), 'TOii')
b(w, out, N, K2, stream=None)
same = (ref == out).all().item()
print(f'RESULT exact: {same}')
if not same:
    print(f'  shapes ref={tuple(ref.shape)} got={tuple(out.shape)}')
    d = (ref != out)
    print(f'  mismatches {d.sum().item()}/{ref.numel()}')
def bench(fn,n=30):
    for _ in range(3): fn()
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t0)/n
t1 = bench(lambda: b(w, out, N, K2, stream=None))
t2 = bench(lambda: w.view(torch.int32).T.contiguous())
print(f'RESULT mine  {t1*1e6:6.1f} us')
print(f'RESULT torch {t2*1e6:6.1f} us  speedup {t2/t1:.1f}x')