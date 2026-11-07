# 24GB 卡：能装多少层？
LAYER_B = 204 * 1024**2
EXPERT_B = 19.7 * 1024**2
CAP = 22 * 2**30   # 4090 可用 22GB
# 方案A：全部 40 层常驻 = 40*204MB = 8.2GB，剩 13.8GB 放专家槽
layers_all = 40 * LAYER_B
rest = CAP - layers_all
slots = int(rest // EXPERT_B)
print(f'24GB 卡：40 层常驻 {layers_all/2**30:.1f}GB，剩 {rest/2**30:.1f}GB -> {slots} 个专家槽')
print(f'每 token 需要 {40*6} 个专家，去重后约 {int(40*6*0.87)} 个')
print(f'槽位 {slots} 个，命中率取决于路由局部性')
print()
# 如果 384 个专家全在一层，那就是 40 层 * 384 = 15360 个，不可能全装
print(f'模型总专家数: {40*384}，总大小 {40*384*EXPERT_B/2**30:.0f}GB')
print(f'24GB 卡最多放 {slots} 个 = {100*slots/(40*384):.2f}% 的专家')
print()
# 每 token 专家传输量（假设 0 命中）
print(f'最坏情况：每 token 传 {240*EXPERT_B/2**30:.2f}GB')
print(f'PCIe 25.5GB/s -> {240*EXPERT_B/25.5e9*1000:.0f} ms')
print()
# 但热专家是稳定的（MoE 有热点）
print('关键：MoE 路由有热点，top 20% 专家通常覆盖 60-80% 访问')
print(f'如果 60% 命中，每 token 传 {240*0.4*EXPERT_B/2**30:.2f}GB -> {240*0.4*EXPERT_B/25.5e9*1000:.0f} ms')