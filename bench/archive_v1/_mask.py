
import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.nn import window_idxs_general, window_idxs_per_row

W = 4
# 两行：行0 在位置 0（只有 1 个 token），行1 在位置 3
pos = torch.tensor([0, 3])
old_len = 0
shared = window_idxs_general(0, 1, old_len, W).unsqueeze(0).expand(2, -1, -1)
per_row = window_idxs_per_row(pos, 1, old_len, W)
print("shared :", shared.tolist())
print("per_row:", per_row.tolist())

# 手工验证：行0 在位置 0，只能看自己 -> [0,-1,-1,-1]
#           行1 在位置 3，看 0..3      -> [0,1,2,3]
