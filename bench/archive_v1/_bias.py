print("=== 分析 got vs ref 的关系 ===")
K = 512
cases = [("all ones", -2304, 1792), ("all 0.5", -1152, 896), ("x[k]=k", -585728, 460672)]
for label, got, ref in cases:
    print(f"{label}: got={got}, ref={ref}")
    print(f"  got/ref = {got/ref:.4f}")
    print(f"  got - ref = {got - ref}")
    print(f"  got + ref = {got + ref}")
    print()
print("all ones: x 全 1，K=512，每 8 个 code 是 0..7")
codes = []
for j in range(K):
    c = (j % 8) & 0xF
    codes.append(c - 16 if c >= 8 else c)
print(f"  sum(codes) = {sum(codes)}")
print(f"  如果 deq 是 -8..7 的偏置：sum(codes+8) = {sum(codes) + 8*K}")
print()
print("观察：got = -2304 = -4.5 x 512")
print("  ref = 1792 = 3.5 x 512")
print("  got/512 = -4.5, ref/512 = 3.5")
print("  差 8 -> 正好是 int4 的 bias！")
print()
print("=== 结论：deq 输出的是 0..15 映射到 -8..7，")
print("     而参考把 nibble 0..7 当作 0..7（不偏置）===")
print()
print("实际上 deq4 的 SUB/MUL/ADD 组合产生的是 signed int4，")
print("即 nibble 0->-8, 1->-7, ..., 7->-1, 8->0, ..., 15->7")
print("这不是 E2M1 的语义！E2M1 应该是: code 0..15 -> {0,0.5,1,1.5,2,3,4,6}")