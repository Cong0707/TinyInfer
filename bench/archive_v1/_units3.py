print("=== 严格单位换算（前几次全错在 MB/TB 上）===")
print()
print("285 MB = 285e6 bytes")
print("9.6 TB/s = 9.6e12 bytes/s")
print("285e6 / 9.6e12 = 2.97e-5 s = 29.7 us  <-- 微秒，不是毫秒！")
print()
per_layer_bytes = 285e6
for n in (2, 3):
    bps = n * 4.8e12
    sec = per_layer_bytes / bps
    us = sec * 1e6
    total_ms = sec * 40 * 1000
    print(f"{n} 卡: {us:.1f} us/层 x 40 层 = {total_ms:.2f} ms -> {1000/total_ms:.0f} tok/s")
print()
print("=== 这才是合理数量级 ===")
print("2 卡: 1.19 ms/token = 841 tok/s")
print("3 卡: 0.79 ms/token = 1262 tok/s")
print()
print("=== 对比笔记本实测 ===")
print("笔记本: 518.8 ms/token = 1.93 tok/s")
print(f"2 卡 H200 比笔记本快 {1190/518.8:.0f} 倍" if False else "")
print()
print("但 H200 上不是纯带宽：40 层 x 每层 ~15 个 kernel = 600 次 launch")
print("CUDA graph 捕获后每次 launch 开销 ~1-2 us，600 x 1.5 = 0.9 ms")
print("-> 和 0.79-1.19 ms 的计算时间同量级，所以真实速度会打折")
print()
print("=== 综合估计（2-3 卡 H200）===")
for n, ms in ((2, 1.19), (3, 0.79)):
    for eff in (0.7, 0.5, 0.35):
        real = ms / eff
        print(f"  {n} 卡, {eff*100:.0f}% 效率: {real:.2f} ms -> {1000/real:.0f} tok/s")
    print()