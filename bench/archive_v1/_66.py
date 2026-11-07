print("=== 66 GB/s 意味着什么 ===")
print()
N, K = 16384, 5120
weights = N * K
GB = weights * 0.5 / 2**30
us = 636.7
print(f"权重 {weights/1e6:.1f}M 个, {GB:.1f} GB fp4")
print(f"耗时 {us:.1f} us -> {GB/(us/1e6):.0f} GB/s")
print()
ips = 26 * 4 * 3090e6     # SM x scheduler x clock
print(f"指令发射 {ips/1e12:.2f} G inst/s")
print()
# time budget
total_cycles = us * 1e-6 * 3090e6
print(f"{us} us = {total_cycles/1e6:.1f}M cycles")
print(f"总指令预算 = {total_cycles * 26 * 4 / 1e6:.1f}M 指令（26 SM x 4 sched）")
per_weight = total_cycles * 26 * 4 / weights
print(f"每权重可用 {per_weight:.1f} 条指令")
print()
print("=== 我的标量解码需要多少条 ===")
print("  e = (c>>1)&3         2")
  m = c&1              1") if False else None
print("  m = c & 1            1")
print("  safe_e = e|(e==0)    2")
print("  bits = ((126+e)<<23)|(m<<22)  3")
print("  uint_as_float        1")
print("  fixup                2")
print("  sign                 2")
print("  fmaf                 1")
print("  ---> 14 条")
print()
print("但一个 warp 32 lane 并行，所以每权重 14/32 = 0.44 条")
print(f"  -> 需要 {weights * 14/32 / 1e9:.2f}G 指令，预算够")
print()
print("=== 所以瓶颈不是指令总数，而是延迟 ===")
print("  每个 lane 串行处理 8 个 nibble，依赖链长")
print("  64 个 warp/SM x 32 lane = 2048 线程，但每线程 8 条依赖链")