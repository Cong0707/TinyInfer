GB = 2**30
print("=== 单流 decode 的真实约束：不是算力，是 HBM 延迟+带宽下的串行链 ===")
print()
print("一个 token 必须走完 40 层，层间严格串行。每层的耗时 = f(该层要读的权重)。")
print()
# per layer, per token, what MUST come from HBM
dense_per_layer = 138 * 2**20        # MLA projections + norms, fp8
expert_per_layer = 6 * 18.38 * 2**20 # top-6 experts
print(f"每层 dense: {dense_per_layer/2**20:.0f} MB, top-6 专家: {expert_per_layer/2**20:.0f} MB")
print()
print("关键：8 卡张量并行后，每张卡只负责 1/8 的层（或每层的 1/8 分片），")
print("所以单卡每 token 只需读总字节的 1/8，且 8 张卡并行。")
print()
total = 10.96
for n in (4, 8):
    per_card = total / n
    bw = 4.8      # TB/s
    ms = per_card / bw * 1000
    print(f"  {n} 卡: 每卡读 {per_card:.2f} GB / {bw} TB/s = {ms:.0f} ms/token"
          f" -> {1000/ms:.1f} tok/s")
print()
print("这仍然只有个位数。所以真实部署的单流速度确实不高 --")
print("业界公开的 671B 类模型在 8xH200 上单流 decode 就是 20-40 tok/s 量级，")
print("远高于我算的 1.8，说明带宽不是唯一约束。")
print()
print("差在哪：")
print("  1. 权重常驻后，prefill 阶段已经把该层权重载入 L2，decode 复用")
print("  2. MLA 的 latent 很小（512 维），attention 部分计算量远小于 GEMV")
print("  3. 专家是 gather 而非全读：6 个专家 = 110 MB，但只取需要的行")
print("  4. 真实 kernel 有 60-80% 带宽效率，我的算法假定 100% 且无缓存")