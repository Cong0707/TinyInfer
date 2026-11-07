import numpy as np
print("=== E2M1 真实值表 ===")
# E2M1: e=2 bits, m=1 bit, bias=1
vals = []
for code in range(16):
    s = -1 if (code & 8) else 1
    e = (code >> 1) & 3
    m = code & 1
    if e == 0:
        v = m * 0.5          # subnormal
    else:
        v = (1 + m/2) * (2 ** (e - 1))
    vals.append(s * v)
print("code: value")
for i, v in enumerate(vals):
    print(f"  {i:2d} (0b{i:04b}): {v:+.2f}")
print()
print("=== 无分支实现思路 ===")
print("把 4 bit 当作: s | e(2b) | m(1b)")
print("可以用 kFp4Lut 常量数组，编译器会放进 constant memory")
print("  constant memory 的广播访问是 1 条指令，和 shared 一样快")
print("  但如果每个 lane 读不同 index，constant memory 会串行化 <- 这是问题")
print()
print("正确做法：用 __ldg 或直接算术重建，避免任何内存查表")
print("算术重建（无分支）：")
print("  e_raw = (code >> 1) & 3")
print("  m = code & 1")
print("  # 2^(e-1) for e>=1, 0.5 for e=0 -> 用 bit trick:")
print("  exp_bits = (e_raw == 0) ? 126 : (126 + e_raw)   # float exponent field")
print("  val = float_from_bits(exp_bits << 23) * (1 + m * 0.5)")
print("  val *= (code & 8) ? -1 : 1")