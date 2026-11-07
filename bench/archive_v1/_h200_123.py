GB = 2**30; MB = 2**20
print("=== 用引擎的真实数据流，算 1-3 张 H200 ===")
print()

# 从 checkpoint 精确统计（之前的实测值）
DENSE_PER_LAYER = 138 * MB      # MLA 投影 + norm + gate + shared
EXPERTS_PER_LAYER = 384 * 18.38 * MB
EXPERT_ONE = 18.38 * MB
TABLES = 3.23 * GB               # embed + head
ENGRAM = 188.83 * GB             # 巨型查表，按行读，不进显存
L = 40
TOPK = 6
print(f"每层: dense {DENSE_PER_LAYER/MB:.0f} MB + {384} 专家 x 18.38 MB = 
      f"{(DENSE_PER_LAYER+EXPERTS_PER_LAYER)/GB:.2f} GB")
print(f"40 层专家总计: {EXPERTS_PER_LAYER*L/GB:.1f} GB")
print(f"40 层 dense : {DENSE_PER_LAYER*L/GB:.1f} GB")
print(f"engram 表   : {ENGRAM/GB:.1f} GB (按行读，不占显存)")
print()

print("=== 引擎的流式机制（已实现）===")
print("  1. flat streaming: 层权重打包成 pinned 缓冲，环深度 BUF_COUNT")
print("  2. expert cache : 按路由结果缓存专家块，14GB 上限")
print("  3. 预取        : ring 深度 2-3，计算当前层时预取下一层")
print()

# 每张 H200
HBM = 141.0
PCIE = 64.0    # GB/s, PCIe 5.0 x16
print("=== 关键：H200 的 PCIe 是 64 GB/s（不是 HBM 的 4.8 TB/s）===")
print("流式路径：RAM -> PCIe -> VRAM，瓶颈在这里")
print()

for n in (1, 2, 3):
    vram = n * HBM
    # dense 全进 VRAM，剩下的放专家
    dense_all = DENSE_PER_LAYER * L / GB
    room_experts = vram - dense_all - TABLES
    total_experts = EXPERTS_PER_LAYER * L / GB
    cache_frac = min(1.0, room_experts / total_experts)
    print(f"=== {n} x H200 = {vram:.0f} GB ===")
    print(f"  dense 全驻留 : {dense_all:6.1f} GB")
    print(f"  tables       : {TABLES:6.1f} GB")
    print(f"  剩余放专家   : {room_experts:6.1f} GB / {total_experts:.1f} GB 
          f"= {100*cache_frac:.0f}%")
    print()
    print(f"   每秒 token 需求（{TOPK} 专家 x {L} 层 = {TOPK*L} 个）：")
    per_token_experts = TOPK * L * EXPERT_ONE / GB
    miss = per_token_experts * (1 - cache_frac)
    print(f"     命中 {cache_frac*100:.0f}% -> 每 token 从 RAM 读 {miss:.2f} GB")
    print(f"     PCIe {PCIE} GB/s -> {miss/PCIE*1000:.0f} ms/token 
          f"= {PCIE/miss:.1f} tok/s")
    print()