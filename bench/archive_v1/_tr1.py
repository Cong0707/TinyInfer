import sys, torch, time
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
# 测：连续 30 次 214MB 传输，看是否有间隔
import tinyinfer.loading as L
n = 214 * 1024**2
src = torch.empty(n, dtype=torch.uint8, pin_memory=True)
dst = torch.empty(n, dtype=torch.uint8, device='cuda')
# 1) 无计算
torch.cuda.synchronize(); t0=time.perf_counter()
for _ in range(28): dst.copy_(src, non_blocking=True)
torch.cuda.synchronize()
t1 = time.perf_counter()-t0
print(f'RESULT 28x214MB 纯传输: {t1*1000:.0f}ms -> {28*n/t1/1e9:.1f} GB/s')
# 2) 中间夹小计算（模拟真实）
a = torch.randn(512, 512, device='cuda')
torch.cuda.synchronize(); t0=time.perf_counter()
for _ in range(28):
    dst.copy_(src, non_blocking=True)
    for _ in range(30): torch.mm(a, a)
torch.cuda.synchronize()
t2 = time.perf_counter()-t0
print(f'RESULT 28x(传输+小计算): {t2*1000:.0f}ms')
# 3) 用 _async_h2d
torch.cuda.synchronize(); t0=time.perf_counter()
for _ in range(28): L._async_h2d(dst, src, n)
torch.cuda.synchronize()
t3 = time.perf_counter()-t0
print(f'RESULT 28x_async_h2d: {t3*1000:.0f}ms -> {28*n/t3/1e9:.1f} GB/s')