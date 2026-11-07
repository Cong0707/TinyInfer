import sys, time, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
import numpy as np

torch.cuda.init()
N = 56 * 1024 * 1024  # 56MB, one expert's fp4 weights

def bench(fn, n=10):
    fn(); torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n

# pageable H2D
cpu = torch.empty(N, dtype=torch.uint8)
gpu = torch.empty(N, dtype=torch.uint8, device="cuda")
ms = bench(lambda: gpu.copy_(cpu, non_blocking=False)) * 1e3
print(f"pageable H2D {N/2**20:.0f}MB: {ms:.2f} ms -> {N/2**30/(ms/1e3):.1f} GB/s")

# pinned H2D
pin = torch.empty(N, dtype=torch.uint8, pin_memory=True)
ms = bench(lambda: gpu.copy_(pin, non_blocking=True)) * 1e3
print(f"pinned   H2D {N/2**20:.0f}MB: {ms:.2f} ms -> {N/2**30/(ms/1e3):.1f} GB/s")

# pinned + stream overlap (copy while computing a matmul)
s = torch.cuda.Stream()
a = torch.randn(4096, 4096, dtype=torch.bfloat16, device="cuda")
w = torch.randn(4096, 4096, dtype=torch.bfloat16, device="cuda")
def overlap():
    with torch.cuda.stream(s):
        gpu.copy_(pin, non_blocking=True)
    torch.matmul(a, w)
    torch.cuda.current_stream().wait_stream(s)
ms = bench(overlap) * 1e3
print(f"overlap copy+matmul: {ms:.2f} ms (matmul alone {bench(lambda: torch.matmul(a,w))*1e3:.2f} ms)")

# RAM->pinned copy bandwidth
ms = bench(lambda: pin.copy_(cpu)) * 1e3
print(f"pageable->pinned {N/2**20:.0f}MB: {ms:.2f} ms -> {N/2**30/(ms/1e3):.1f} GB/s")
