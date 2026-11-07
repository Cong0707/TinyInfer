print("=== E2M1 vs int4 语义对比 ===")
print()
e2m1 = []
for code in range(16):
    s = -1 if (code & 8) else 1
    e = (code >> 1) & 3
    m = code & 1
    v = (m * 0.5) if e == 0 else (1 + m / 2) * (2 ** (e - 1))
    e2m1.append(s * v)

print(f"{'code':>4} {'int4':>6} {'E2M1':>7}")
for c in range(16):
    i4 = c - 16 if c >= 8 else c
    print(f"{c:4d} {i4:6d} {e2m1[c]:7.2f}")
print()
print("=== 关键：E2M1 的值是 0.5 的整数倍 ===")
print("  {0, 0.5, 1, 1.5, 2, 3, 4, 6} x 2 = {0,1,2,3,4,6,8,12} 全是整数")
print("  可以用 int8 mma 计算，最后除以 2")
print()
print("=== 正确的解码路径 ===")
print("  1. 用 lop3 快速提取 nibble（这一步仍然有效）")
print("  2. 但映射表必须是 E2M1 的 16 个值，不是 int4 线性映射")
print("  3. 查表 -> 16 项常量，编译器放 constant memory")
print()
print("  或者：用 __byte_perm + LUT 一次处理 4 个 nibble")