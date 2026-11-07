print("=== 66 GB/s 意味着什么 ===")
N, K = 16384, 5120
weights = N * K
GB = weights * 0.5 / 2**30
us = 636.7
print(f"权重 {weights/1e6:.1f}M 个, {GB:.1f} GB fp4")
print(f"耗时 {us:.1f} us -> {GB/(us/1e6):.0f} GB/s")
print()
total_cycles = us * 1e-6 * 3090e6
budget = total_cycles * 26 * 4
print(f"{us} us = {total_cycles/1e6:.1f}M cycles")
print(f"指令预算 = {budget/1e6:.1f}M（26 SM x 4 sched x cycles）")
per_weight = budget / weights
print(f"每权重可用 {per_weight:.2f} 条指令")
print()
print("标量解码需要 14 条/权重（单 lane），但 32 lane 并行 -> 0.44 条/权重")
print("所以指令数不是瓶颈")
print()
print("=== 真正的瓶颈：每个 lane 的 8 个 nibble 是串行依赖 ===")
print("  p = fmaf(a[k0+0], fp4_fast(n0), p)")
print("  p = fmaf(a[k0+1], fp4_fast(n1), p)   <- 依赖上一个 p")
print("  ... 8 次，形成 8 级 FMA 依赖链")
print("  每级延迟 ~4 cycle -> 32 cycles 才能完成 8 个权重")
print()
print("  32 lane 并行 = 32 个这样的链")
print(f"  每个 warp 每次迭代处理 {32*8} 个权重，需要 32 cycles")
print(f"  26 SM x 4 sched x 64 warps 并发...")
print()
print("=== 解决：用 4 个独立的累加器打破依赖链 ===")
print("  p0, p1, p2, p3 各累加 2 个 nibble，最后相加")
print("  依赖链从 8 级降到 2 级")