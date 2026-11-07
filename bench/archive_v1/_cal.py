print("=== 效率校准 ===")
print()
print("笔记本实测: 518.8 ms/token, 1.93 tok/s")
print("  每 token 从页缓存+磁盘读 5.59 GB (10 层驻留)")
print("  有效读取 10.8 GB/s vs 磁盘 2.85 GB/s -> 页缓存帮了 3.8 倍")
print()
gpu_bytes = 10 * 178e6
eff = gpu_bytes / 0.5188
print(f"笔记本 GPU 侧实际处理: 10 层 x 178 MB / 0.5188s = {eff/1e9:.1f} GB/s")
print(f"  峰值 336 GB/s -> 利用率 {eff/336e9*100:.0f}%")
print("  (低是因为 GPU 大部分时间在等磁盘)")
print()
print("H200 全部常驻 HBM，不存在等待，利用率会高很多。")
print("但 M=1 GEMV + 40 层串行仍有损耗，取 45-65%。")
print()
BYTES = 9.24e9
for n in (2, 3):
    bw = n * 4.8e12
    ideal = bw / BYTES
    print(f"{n} 卡: 理想 {ideal:.0f} tok/s")
    for e in (0.65, 0.55, 0.45):
        print(f"  {e*100:.0f}%: {ideal*e:.0f} tok/s")
    print()