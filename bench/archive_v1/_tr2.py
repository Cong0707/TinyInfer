import sys, torch, time
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
# 专家传输：240 次 18.8MB 小传输的效率
n = 18800640  # 18.8MB
src = torch.empty(n, dtype=torch.uint8, pin_memory=True)
dst = torch.empty(n, dtype=torch.uint8, device='cuda')
torch.cuda.synchronize(); t0=time.perf_counter()
for _ in range(240): dst.copy_(src, non_blocking=True)
torch.cuda.synchronize()
t = time.perf_counter()-t0
print(f'RESULT 240x18.8MB: {t*1000:.0f}ms -> {240*n/t/1e9:.1f} GB/s ({t/240*1e6:.0f}us each)')
# 对比：一次传 240 个的量
big = torch.empty(240*n, dtype=torch.uint8, pin_memory=True)
bdst = torch.empty(240*n, dtype=torch.uint8, device='cuda')
torch.cuda.synchronize(); t0=time.perf_counter()
bdst.copy_(big, non_blocking=True)
torch.cuda.synchronize()
t2 = time.perf_counter()-t0
print(f'RESULT 1x4.5GB: {t2*1000:.0f}ms -> {240*n/t2/1e9:.1f} GB/s')
print(f'RESULT 小传输开销: {t/t2:.2f}x')