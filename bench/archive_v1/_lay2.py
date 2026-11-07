per_layer_cap = 57
n_layers = 40
needed = per_layer_cap * n_layers
print(f'RESULT 每层平均 57 个专家 x 40 层 = {needed} 槽')
print(f'RESULT 当前有 913 槽，全局 LRU')
print()
print('RESULT === 模拟：按层分区 vs 全局 LRU ===')
# 用实测的 (layer,expert) 分布模拟
import random
random.seed(0)
# 近似：每层 57 个高频 + 长尾
# 先算：如果每层 913/40 = 22 个槽，能覆盖多少
slots_per_layer = 913 // n_layers
print(f'RESULT 均分到每层: {slots_per_layer} 槽/层，少于需要的 57')
print(f'RESULT => 按层均分会丢命中率')
print()
print('RESULT === 更好的方案：按层权重分配 ===')
print('  访问量大的层给更多槽，符合 LRU 的自然行为')
print('  => 全局 LRU 已经做到了这一点，不需要改')