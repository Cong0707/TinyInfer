print("=== 正确的 decode 模型：不是带宽除法，是 kernel 串行链 ===")
print()
print("一个 token 走 40 层，层间串行。每层的耗时 = 该层 kernel 的执行时间。")
print("每层 kernel 时间 = max(该层要读的权重 / 有效带宽, 该层 FLOPs / 算力)")
print()
GB = 2**30; MB = 2**20
DENSE_L = 175 * MB
EXP_TOK = 6 * 18.38 * MB   # top-6 for one token
L = 40
print(f"每层: dense {DENSE_L/MB:.0f} MB + top-6 专家 {EXP_TOK/MB:.0f} MB = {(DENSE_L+EXP_TOK)/MB:.0f} MB")
print(f"40 层串行 = {(DENSE_L+EXP_TOK)*L/GB:.2f} GB  (和之前一致)")
print()

TBS = 4.8
print("=== 单卡 vs 多卡：关键在【层怎么分】 ===")
print()
print("张量并行（TP）：每层的权重切成 N 份，N 卡同时算同一层")
print("  -> 每卡每层读 (DENSE_L+EXP_TOK)/N，N 卡并行 => 等效带宽 N 倍")
print()
for n in (1, 2, 3):
    per_card_layer = (DENSE_L + EXP_TOK) / n / MB
    ms_per_layer = per_card_layer * MB / (TBS * 1e12) * 1000
    total_ms = ms_per_layer * L
    print(f"{n} 卡 TP: 每卡每层 {per_card_layer:.0f} MB, {ms_per_layer*1000:.0f} us/层")
    print(f"        {L} 层 = {total_ms:.0f} ms -> {1000/total_ms:.1f} tok/s")
print()
print("这和之前算的一样，因为带宽确实随卡数线性增长。")
print()
print("=== 那 1000 tok/s 从哪来？ ===")
print()
print("重读那个公开数据：")
print("  『8块H200 NVL 部署的 671B DeepSeek R1，最高达到 1000 tokens/s 的平均生成速度』")
print()
print("关键词是【平均生成速度】，不是单流。1000 tok/s 是 batch 32 时的【总】吞吐。")
print("折算到单流：1000 / 32 ≈ 31 tok/s —— 这个数字就合理了！")
print()
print("验证：")
for n in (2, 3):
    bw = n * TBS
    per_tok = (DENSE_L + EXP_TOK) * L / GB
    ms = per_tok / bw * 1000
    single = 1000/ms
    print(f"  {n} 卡: 单流 {single:.1f} tok/s；batch 32 总吞吐约 {single*32:.0f} tok/s")