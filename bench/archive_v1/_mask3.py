
import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.nn import window_idxs_per_row

# 真实场景：两行在同一个 KV 缓冲里，但各自只有自己的历史
# 行0: prompt 1 token, 现在位置 0, 历史 0 -> kv_att 长度 1 (只有它自己)
# 行1: prompt 5 token, 现在位置 4, 历史 4 -> kv_att 长度 5
# 同一个 batch 里两行的 old_len 不同 -> 不能共用一个 kv_att 形状
W = 4
pos = torch.tensor([0, 4])
for old in (0, 4):
    g = window_idxs_per_row(pos, 1, old, W)
    print(f"old_len={old}:", g.tolist())
