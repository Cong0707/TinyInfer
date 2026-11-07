GB = 2**30
print("=== 修正单位错误：GB / TB/s = 毫秒（不是秒）===")
print()
PER_TOKEN = 11.14   # GB
print(f"每 token 需读 {PER_TOKEN} GB（40 层全算 + 240 专家 + head）")
print()
for n in (1, 2, 3, 4, 8):
    bw = n * 4.8          # TB/s
    ms = PER_TOKEN / bw * 1000
    print(f"{n} x H200 ({bw:4.1f} TB/s): {ms:6.2f} ms/token -> {1000/ms:6.0f} tok/s")
print()
print("=== 但这些是纯带宽理想值，40 层串行的 kernel 开销呢 ===")
print()
print("每层耗时 = 该层字节 / (卡数 x 4.8 TB/s)")
for n in (2, 3):
    per_layer = (175 + 110) / 1024  # GB
    us = per_layer / (n*4.8) * 1e6
    print(f"  {n} 卡: 每层 {per_layer*1024:.0f} MB / {n*4.8:.1f} TB/s = {us:.1f} us")
    print(f"       40 层 = {us*40/1000:.2f} ms -> {1e6/(us*40):.0f} tok/s")
    print(f"       若每层再加 5 us kernel 开销: {us+5:.1f} us -> {1e6/((us+5)*40):.0f} tok/s")
print()
print("=== 结论：2-3 卡的合理区间 ===")
for n in (2, 3):
    bw_ms = PER_TOKEN / (n*4.8) * 1000
    ideal = 1000/bw_ms
    realistic = ideal * 0.5    # 50% efficiency: launch overhead, sync, TP communication
    print(f"  {n} 卡: 理想 {ideal:.0f} tok/s, 实际 50% 效率 {realistic:.0f} tok/s (单流)")
print()
print("批量吞吐（权重共享, FLOPs 主导）：")
ACTIVE = 37e9
for n in (2, 3):
    flops = n * 1979e12
    ceil = flops / (2*ACTIVE)
    print(f"  {n} 卡算力上限 {ceil:.0f} tok/s；批量 8-32 时总吞吐可达 {ceil*0.3:.0f}-{ceil*0.5:.0f} tok/s")