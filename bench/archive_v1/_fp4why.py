"""Why is gemv_fp4 only 4% of peak? Check launch config and access pattern."""
import sys
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
import torch
import tinyinfer.torch_compat  # noqa: F401
from tinyinfer import native

GB = 2**30
N, K = 4096, 4096
Wq = torch.randint(0, 256, (N, K//2), dtype=torch.uint8, device="cuda")
sc = torch.full((N, K//32), 127, dtype=torch.uint8, device="cuda")
a = torch.randn(K, dtype=torch.float32, device="cuda")
asc = torch.ones(K//32, dtype=torch.float32, device="cuda")
y = torch.zeros(N, dtype=torch.bfloat16, device="cuda")

def bench(grid, block, n=100):
    with torch.inference_mode():
        for _ in range(10):
            native.launch(native.kernel("gemv_fp4_e8m0", "gemv_fp4_e8m0"),
                          grid, block, [a, asc, Wq, sc, y, N, K])
        torch.cuda.synchronize()
        e0 = torch.cuda.Event(True); e1 = torch.cuda.Event(True)
        e0.record()
        for _ in range(n):
            native.launch(native.kernel("gemv_fp4_e8m0", "gemv_fp4_e8m0"),
                          grid, block, [a, asc, Wq, sc, y, N, K])
        e1.record(); torch.cuda.synchronize()
    ms = e0.elapsed_time(e1)/n
    by = N * K * 0.5
    return ms, by/(ms/1000)/1e9

# kernel hardcodes 16 rows per block, 512 threads (r = tid>>5, so 16 warps)
print("kernel: blockIdx.x * 16 + warp, 512 threads -> 16 warps")
print(f"N={N}, grid = ceil(N/16) = {(N+15)//16}")
ms, gbs = bench(((N+15)//16,), (512,))
print(f"  {ms*1000:.1f} us, {gbs:.0f} GB/s")
print()
print("problem: each warp reads a full row of fp4 (K/2 = 2048 bytes),")
print("but each lane reads only ONE byte per 32-k block -> terrible coalescing?")
print()
# check: what does a warp's access look like?
print("inner loop: for kb in range(nb): for j in lane<32 step 32: read w[n*khalf + k>>1]")
print("  -> lane j reads byte at offset (kb*32+j)>>1")
print("  -> 32 lanes read 32 consecutive k values = 16 consecutive bytes")
print("  -> only 16 bytes per warp transaction! (128-byte cache line is 8x wasted)")
