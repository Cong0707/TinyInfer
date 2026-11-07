"""Can we decode 8 nibbles from one 32-bit word with fewer instructions?"""
import sys
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
import torch
print("=== E2M1 值的规律（找 bit trick）===")
vals = []
for code in range(16):
    s = -1 if (code & 8) else 1
    e = (code >> 1) & 3
    m = code & 1
    v = (m * 0.5) if e == 0 else (1 + m/2) * (2 ** (e - 1))
    vals.append(s * v)
for i in range(16):
    print(f"  {i:2d} 0b{i:04b} -> {vals[i]:+5.2f}   exponent_if_normal={((i>>1)&3)}")
print()
print("观察：e=0 -> 0 或 0.5（次正规）")
print("      e=1 -> 1, 1.5")
print("      e=2 -> 2, 3")
print("      e=3 -> 4, 6")
print()
print("=> 值 = 2^(e-1) * (1 + m/2)，e=0 时特例")
print("   等价于 float 位模式: exponent = 127 + (e-1) = 126+e，mantissa 高位 = m<<22")
print()
print("=== 关键：能不能用一次 32-bit 位运算批量转换？ ===")
print("8 个 nibble 在一个 word 里。如果 e 和 m 分别提取：")
print("  e_bits = (word >> 1) & 0x77777777   # 每 nibble 的 bit1-2")
print("  m_bits = word & 0x11111111          # 每 nibble 的 bit0")
print("  s_bits = word & 0x88888888")
print()
print("然后 (126+e)<<23 需要每个 nibble 独立处理 -> 还是得展开")
print("但可以用 __byte_perm / PRMT 指令批量打包")
print()
print("实测路径：先把 lookup 表放进 __constant__，看是否比算术快")
