import sys, torch, time
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.kernels import quant as Q
from tinyinfer.kernels import native
K = 5120
x = torch.randn(1, K, dtype=torch.bfloat16, device='cuda')
def bench(fn,n=2000):
    for _ in range(50): fn()
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t0)/n*1e6
print(f'RESULT act_quant       : {bench(lambda: Q.act_quant(x, 32, "ue8m0", torch.float8_e8m0fnu)):.2f} us')
print(f'RESULT _native_ok      : {bench(Q._native_ok):.2f} us')
print(f'RESULT contiguous      : {bench(lambda: x.contiguous()):.2f} us')
print(f'RESULT view(1,K)       : {bench(lambda: x.view(1, K)):.2f} us')
print(f'RESULT new_empty scale : {bench(lambda: x.new_empty(1, K//32, dtype=torch.float8_e8m0fnu)):.2f} us')
print(f'RESULT empty fp8       : {bench(lambda: torch.empty((1, K), dtype=torch.float8_e4m3fn, device=x.device)):.2f} us')
z = x.view(1, K)
s = x.new_empty(1, K//32, dtype=torch.float8_e8m0fnu)
y = torch.empty((1, K), dtype=torch.float8_e4m3fn, device=x.device)
print(f'RESULT _launch (bound) : {bench(lambda: Q._launch(z, y, s, "act_quant")):.2f} us')