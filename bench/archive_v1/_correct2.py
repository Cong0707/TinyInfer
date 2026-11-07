GB = 2**30; MB = 2**20
print("=== 修正：反量化在 kernel 内部，不产生额外读取 ===")
print()
print("gemv_fp4_e8m0 的工作方式：")
print("  1. 从 HBM 读 fp4 原始字节 (每权重 0.5 字节)")
print("  2. 拆 nibble，查 16 项常量表得 float")
print("  3. 尺度字节 e8m0 内联解码 (1 次 ldexpf)")
print("  -> 读取量 = fp4 原始大小，没有 4 倍膨胀")
print()
DENSE = 6.84; HEAD = 1.26
EXP_FP4 = 4.31      # 240 experts at fp4 native size
total = DENSE + EXP_FP4 + HEAD
print(f"每 token 读取: dense {DENSE} + experts {EXP_FP4} + head {HEAD} = {total:.2f} GB")
print()
print("=== H200 3 卡的速度 ===")
BW = 3 * 4.8   # TB/s
ms = total / BW * 1000
print(f"  {total:.2f} GB / {BW:.1f} TB/s = {ms:.1f} ms -> {1000/ms:.0f} tok/s")
print()
print("=== 但这仍是纯 HBM 带宽下限。真实的 H200 会快得多，因为： ===")
print()
print("1) 专家缓存命中后不用重读")
print("   H200 3 卡 423 GB 显存，专家池只有 275.7 GB -> 100% 常驻")
print("   命中率接近 100% 时，专家权重留在 L2/HBM 原地，不重复搬运")
print()
print("2) L2 缓存 60 MB/卡 x 3 = 180 MB")
print(f"   每层 dense 只有 175 MB，几乎整个能进 L2")
print("   40 层轮流用 L2，实际 HBM 流量远低于 6.84 GB")
print()
print("3) 真正的瓶颈变成 kernel 执行时间，不是带宽")
print()
print("=== 用算力估算 ===")
ACTIVE = 37e9
flops_per_tok = 2 * ACTIVE * 2  # fp4 也算 2 FLOP/param (madd)
print(f"激活 37B 参数 x 2 FLOP = {flops_per_tok/1e12:.2f} TFLOP/token")
print()
print("H200 fp8 稠密算力 989 TFLOPS (1979 稀疏)")
print("fp4 权重在 H200 上按 fp8 算：")
for n in (2, 3):
    fp8 = n * 989e12
    tps = fp8 / flops_per_tok
    print(f"  {n} 卡: {fp8/1e12:.0f} TFLOP/s / {flops_per_tok/1e12:.2f} TFLOP = {tps:.0f} tok/s")