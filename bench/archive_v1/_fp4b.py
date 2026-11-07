print("=== 修正单位：GB / (TB/s) = 毫秒 ===")
print()
DENSE = 6.84; HEAD = 1.26
cases = (("Blackwell fp4", 4.31), ("H200 fp8", 8.62), ("H200 bf16", 17.23))
print("每 token 读取量:")
for name, exp in cases:
    print(f"  {name:16s}: {DENSE:.2f} + {exp:5.2f} + {HEAD:.2f} = {DENSE+exp+HEAD:6.2f} GB")
print()
for n in (2, 3, 4):
    bw = n * 4.8
    print(f"=== {n} x H200 / B200, {bw:.1f} TB/s ===")
    for name, exp in cases:
        tot = DENSE + exp + HEAD
        ms = tot / bw * 1000
        print(f"  {name:16s}: {ms:6.1f} ms -> {1000/ms:6.0f} tok/s")
    print()
print("=== B200 原生 fp4 还有额外优势 ===")
print("  同样 3 卡: B200 读 12.41 GB, H200(fp8 反量化) 读 16.72 GB")
print(f"  差 {16.72/12.41:.2f} 倍 -> H200 慢 35%")
print()
print("  而且 B200 带宽 8 TB/s/card（H200 是 4.8），3 卡 = 24 TB/s")
for exp, label in ((4.31, "B200 fp4"), (8.62, "H200 fp8")):
    tot = DENSE + exp + HEAD
    for n, bw in ((3, 3*8.0), (3, 3*4.8)):
        ms = tot / bw * 1000
        card = "B200" if bw > 20 else "H200"
        print(f"  {card} {n}卡: {tot:.2f} GB / {bw:.1f} TB/s = {ms:.1f} ms -> {1000/ms:.0f} tok/s")
    break