import sys, torch, time
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
torch.cuda.init(); torch.zeros(1, device='cuda')
SZ = 178*2**20
host = torch.empty(SZ, dtype=torch.uint8, pin_memory=True)
dev = torch.empty(SZ, dtype=torch.uint8, device='cuda')
def bench(fn, n=10):
    for _ in range(3): fn()
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t0)/n
t1 = bench(lambda: dev.copy_(host, non_blocking=True))
print('RESULT pinned->dev copy_: %.2f ms  %.1f GB/s' % (t1*1000, SZ/t1/1e9))
# 分块拷贝（模拟现在的逐 tensor）
chunks = [(i*6*2**20, min(6*2**20, SZ-i*6*2**20)) for i in range(30) if i*6*2**20 < SZ]
def per_tensor():
    for off, n in chunks: dev[off:off+n].copy_(host[off:off+n], non_blocking=True)
t2 = bench(per_tensor)
print('RESULT 30 x 6MB chunks : %.2f ms  %.1f GB/s' % (t2*1000, SZ/t2/1e9))
print('RESULT 分块开销         : %.2f ms (%.0f%%)' % ((t2-t1)*1000, (t2/t1-1)*100))