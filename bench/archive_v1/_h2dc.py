import sys, torch, time
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
import ctypes
torch.cuda.init(); torch.zeros(1, device='cuda')
# 测 pinned -> device 的 copy_ 在不同大小下的耗时
for mb in (0.1, 0.3, 1, 4, 18.8):
    n = int(mb*2**20)
    host = torch.empty(n, dtype=torch.uint8, pin_memory=True)
    dev = torch.empty(n, dtype=torch.uint8, device='cuda')
    for _ in range(5): dev.copy_(host, non_blocking=True)
    torch.cuda.synchronize()
    N = 200 if mb < 2 else 50
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(N): dev.copy_(host, non_blocking=True)
    torch.cuda.synchronize(); dt=(time.perf_counter()-t0)/N
    print(f'RESULT {mb:5.1f} MB: {dt*1e6:7.1f} us  {n/dt/1e9:6.2f} GB/s  fixed={(dt-n/28.5e9)*1e6:6.1f} us')
    del host, dev; torch.cuda.empty_cache()