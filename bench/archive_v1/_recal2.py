print("=== 严格单位（这次用 Python 直接算）===")
print()
OFFICIAL = 184.3
ms = 1000 / OFFICIAL
print(f"官方单流 {OFFICIAL} tok/s = {ms:.2f} ms/token")
print()
# my worst-case estimate: 9.24 GB/token of weights
W = 9.24e9
print(f"我的最坏情况: 每 token 读 {W/1e9:.2f} GB 权重")
print()
for n in (2, 3, 4, 8):
    bw = n * 4.8e12            # bytes/s
    per_card = W / n            # TP: each card reads 1/n
    sec_per_card = per_card / 4.8e12   # its own share over its own bandwidth
    # cards work in parallel on the same layer, so the step time = per_card / per-card bw
    ms_worst = sec_per_card * 1000
    print(f"{n} 卡 TP: 每卡 {per_card/1e9:.2f} GB / 4.8 TB/s = {ms_worst:.2f} ms -> {1000/ms_worst:.0f} tok/s (最坏情况)")
print()
print("官方 8 卡是 5.43 ms/token。我算的最坏情况 8 卡是:")
per_card8 = W / 8
ms8 = per_card8 / 4.8e12 * 1000
print(f"  {ms8:.2f} ms -> {1000/ms8:.0f} tok/s")
print(f"  官方 {ms:.2f} ms -> {OFFICIAL} tok/s")
print(f"  我快 {ms/ms8:.1f} 倍，说明我的读取量高估了 {ms/ms8:.1f} 倍")
print()
print("=== 用官方数据校正系数 ===")
factor = ms / ms8
print(f"校正系数 = {factor:.1f}（我高估的倍数）")
print(f"真实每 token 读取约 {W/1e9/factor:.2f} GB")
print()
print("=== 2-3 卡的预测（用校正后的量）===")
W_real = W / factor
for n in (2, 3):
    per_card = W_real / n
    ms_n = per_card / 4.8e12 * 1000
    print(f"{n} 卡: 每卡读 {per_card/1e9:.2f} GB -> {ms_n:.2f} ms -> {1000/ms_n:.0f} tok/s")