import sys, torch, time
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.loading import _async_h2d
# 模拟：28 个小 tensor 逐个 DMA vs 先拼接到连续 buffer 再一次 DMA
sizes = [11*1024]*18 + [40*1024*1024, 40*1024*1024, 32*1024*1024] + [5*1024*1024]*7
total = sum(sizes)
print(f'RESULT total {total/2**20:.1f}MB, {len(sizes)} tensors')
# 源：各自独立的 pinned tensor
srcs = [torch.empty(s, dtype=torch.uint8, pin_memory=True) for s in sizes]
dst = torch.empty(total, dtype=torch.uint8, device='cuda')
# 方式1：逐个
def by_one():
    off = 0
    for s, src in zip(sizes, srcs):
        _async_h2d(dst[off:off+s], src, s)
        off += s
torch.cuda.synchronize(); t0=time.perf_counter()
for _ in range(10): by_one()
torch.cuda.synchronize(); t1=(time.perf_counter()-t0)/10
# 方式2：先拼接到连续 pinned，再一次 DMA
stage = torch.empty(total, dtype=torch.uint8, pin_memory=True)
def coalesced():
    off = 0
    for s, src in zip(sizes, srcs):
        stage[off:off+s].copy_(src)
        off += s
    _async_h2d(dst, stage, total)
torch.cuda.synchronize(); t0=time.perf_counter()
for _ in range(10): coalesced()
torch.cuda.synchronize(); t2=(time.perf_counter()-t0)/10
print(f'RESULT by_one    {t1*1000:.1f}ms')
print(f'RESULT coalesced {t2*1000:.1f}ms  ({t1/t2:.2f}x)')