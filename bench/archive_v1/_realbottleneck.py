GB = 2**30
print("=== 真实瓶颈：40 层串行 x 每层 kernel 链 ===")
print()
# measured on my laptop: one decode step of Qwen 0.8B (24 layers) in a CUDA graph
print("实测参照（我机器，Qwen 0.8B，24 层，已捕获 CUDA graph）：")
print("  5.54 ms/token = 180 tok/s")
print("  每层 = 5.54 / 24 = 0.231 ms")
print()
print("但 Qwen 0.8B 每层权重只有 41 MB，而 DeepSeek 每层是：")
DENSE = 175/1024   # GB
EXP = 6*18.38/1024 # GB per token
print(f"  dense {DENSE*1024:.0f} MB + top6 专家 {EXP*1024:.0f} MB = {(DENSE+EXP)*1024:.0f} MB")
print(f"  是 Qwen 单层的 {(DENSE+EXP)*1024/41:.1f} 倍")
print()
print("=== H200 上每层要多久 ===")
per_layer = DENSE + EXP
for n in (2, 3):
    bw = n * 4.8
    ms = per_layer / bw * 1000
    print(f"  {n} 卡: {per_layer*1024:.0f} MB / {bw:.1f} TB/s = {ms:.3f} ms/层")
    print(f"       40 层 = {ms*40:.2f} ms -> {1000/(ms*40):.0f} tok/s")
print()
print("=== 但 L2 和专家常驻会大幅改善 ===")
print()
print("专家 100% 常驻后，240 个专家不需要重新加载，只是从 HBM 读一次算一次。")
print("L2 180 MB 能装下整个 dense 层（175 MB），所以 dense 部分命中率极高。")
print()
print("保守估计：有效带宽按 HBM 的 70% 利用率，且有 50% 的数据从 L2 来")
for n in (2, 3):
    bw = n * 4.8 * 0.7 * 1.5   # 70% efficiency, 1.5x benefit from L2 hits
    ms = per_layer / bw * 1000
    print(f"  {n} 卡: 有效 {bw:.1f} TB/s -> {ms:.3f} ms/层 -> {1000/(ms*40):.0f} tok/s")