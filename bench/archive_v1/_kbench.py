"""Measure each hand-written kernel's achieved bandwidth / FLOPs on this card.

These ratios are the calibration anchor: they say how close the kernels get to the
hardware peak, which is the part that transfers to another GPU (the peaks themselves don't).
"""
import sys, time
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
import torch
import tinyinfer.torch_compat  # noqa: F401
from tinyinfer import native

GB = 2**30; MB = 2**20
# card peaks (this GPU): ~336 GB/s DRAM at 12001 MHz over 128-bit,
# bf16 tensor 36.9 TFLOPS (fp8 ~2x)
PEAK_BW = 336e9
PEAK_BF16 = 36.9e12

def bench(fn, n=200, warm=30):
    with torch.inference_mode():
        for _ in range(warm): fn()
        torch.cuda.synchronize()
        e0 = torch.cuda.Event(True); e1 = torch.cuda.Event(True)
        e0.record()
        for _ in range(n): fn()
        e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1) / n

results = []

# 1) dense bf16 GEMV (M=1) -- the core decode op
for N, K in ((4096, 4096), (6144, 1024), (248320, 1024)):
    x = torch.randn(K, dtype=torch.bfloat16, device="cuda")
    W = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
    y = torch.zeros(N, dtype=torch.float32, device="cuda")
    ms = bench(lambda: native.launch(native.kernel("gemv_bf16", "gemv_bf16"),
                                     ((N + 7)//8,), (256,), [x, W, y, N, K]))
    by = N * K * 2
    bw = by / (ms / 1000)
    results.append(("gemv_bf16", f"{N}x{K}", ms*1000, bw/1e9, bw/PEAK_BW*100))

# 2) fp4 GEMV -- the expert kernel
for N, K in ((4096, 4096), (2304*2, 5120)):
    Wq = torch.randint(0, 256, (N, K//2), dtype=torch.uint8, device="cuda")
    sc = torch.full((N, K//32), 127, dtype=torch.uint8, device="cuda")
    a = torch.randn(K, dtype=torch.float32, device="cuda")
    asc = torch.ones(K//32, dtype=torch.float32, device="cuda")
    y = torch.zeros(N, dtype=torch.bfloat16, device="cuda")
    try:
        ms = bench(lambda: native.launch(native.kernel("gemv_fp4_e8m0", "gemv_fp4_e8m0"),
                                         ((N + 15)//16,), (512,), [a, asc, Wq, sc, y, N, K]))
        by = N * K * 0.5                      # fp4 bytes read
        bw = by / (ms / 1000)
        flops = 2 * N * K / (ms / 1000)
        results.append(("gemv_fp4", f"{N}x{K}", ms*1000, bw/1e9, bw/PEAK_BW*100))
    except Exception as e:
        results.append(("gemv_fp4", f"{N}x{K}", float('nan'), 0, 0))

print(f"card peak: {PEAK_BW/1e9:.0f} GB/s DRAM, {PEAK_BF16/1e12:.1f} TFLOPS bf16")
print()
print(f"{'kernel':14s} {'shape':>14s} {'us':>8s} {'GB/s':>8s} {'%peak':>7s}")
for name, shape, us, gbs, pct in results:
    print(f"{name:14s} {shape:>14s} {us:8.1f} {gbs:8.0f} {pct:6.0f}%")
