
import torch, time
torch.cuda.init()
# 1 GB H2D 大块拷贝，pinned
n = 256 * 1024 * 1024            # 256M floats = 1 GiB
host = torch.empty(n, dtype=torch.uint8, pin_memory=True)
dev  = torch.empty(n, dtype=torch.uint8, device="cuda")
host.fill_(1)
torch.cuda.synchronize()
for _ in range(3):
    dev.copy_(host, non_blocking=True)
torch.cuda.synchronize()
R = 10
t0 = time.perf_counter()
for _ in range(R):
    dev.copy_(host, non_blocking=True)
torch.cuda.synchronize()
dt = time.perf_counter() - t0
gb = n / 1e9
print(f"H2D pinned  {gb*R/dt:.1f} GB/s   ({gb:.2f} GiB x {R} in {dt*1000:.0f} ms)")

# 小分片（模拟专家 18.8MB）看逐片开销
chunk = 18_800_640
for pieces in (1,):
    h2 = torch.empty(chunk, dtype=torch.uint8, pin_memory=True); h2.fill_(1)
    d2 = torch.empty(chunk, dtype=torch.uint8, device="cuda")
    d2.copy_(h2, non_blocking=True); torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(100): d2.copy_(h2, non_blocking=True)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    print(f"H2D 18.8MB  {chunk*100/dt/1e9:.1f} GB/s  ({chunk/1e6:.1f} MB each, 100 pcs)")
