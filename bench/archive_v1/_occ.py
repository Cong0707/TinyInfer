import sys
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
import torch
import tinyinfer.torch_compat  # noqa: F401
from tinyinfer import native

N, K = 4096, 4096
Wq = torch.randint(0, 256, (N, K//2), dtype=torch.uint8, device="cuda")
sc = torch.full((N, K//32), 127, dtype=torch.uint8, device="cuda")
a = torch.randn(K, dtype=torch.float32, device="cuda")
asc = torch.ones(K//32, dtype=torch.float32, device="cuda")
y3 = torch.zeros(N, dtype=torch.bfloat16, device="cuda")
by = N*K*0.5
print(f"total fp4 bytes: {by/2**20:.1f} MB, rows {N}, K {K}")

def bench(grid, block, shared):
    with torch.inference_mode():
        for _ in range(20):
            native.launch(native.kernel("gemv_fp4_e8m0","gemv_fp4_e8m0_v3"),
                          grid, block, [a,asc,Wq,sc,y3,N,K], shared=shared)
        torch.cuda.synchronize()
        e0 = torch.cuda.Event(True); e1 = torch.cuda.Event(True)
        e0.record()
        for _ in range(100):
            native.launch(native.kernel("gemv_fp4_e8m0","gemv_fp4_e8m0_v3"),
                          grid, block, [a,asc,Wq,sc,y3,N,K], shared=shared)
        e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1)/100

# v3 hardcodes 8 rows/block, 256 threads
ms = bench(((N+7)//8,), (256,), K*4)
print(f"v3 8 rows/block, 256 threads: {ms*1000:.1f} us, {by/(ms/1000)/1e9:.0f} GB/s")
print()
print(f"blocks = {(N+7)//8}, each block reads 8 rows x {K//2} bytes = {8*K//2/1024:.0f} KB")
print(f"SM count on this card: 26 -> {((N+7)//8)/26:.1f} blocks per SM")
print()
print("per-warp work: one row = 2KB, with 32 lanes x 4-byte loads = 16 iterations")
print("  -> each warp does only 16 loads then finishes. Latency-bound, not bandwidth.")
print()
print("fix: give each warp MANY rows so loads pipeline. Try 32 rows/block.")
