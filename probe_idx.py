import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.nn import window_idxs_per_row

W = 128
# 单行：位置 1，写了 1 个 token 到槽位 1
idx = window_idxs_per_row(torch.tensor([1]), 1, [1], W, "cpu")
print("single pos1 idx[-1] =", idx[0, 0, -1].item(), flush=True)
print("  -> should read slot 1 (one token written at 1 % 128 = 1)", flush=True)

# 两行：行0 位置 1、行1 位置 2
idx2 = window_idxs_per_row(torch.tensor([1, 2]), 1, [1, 2], W, "cpu")
print("row0 idx[-1] =", idx2[0, 0, -1].item(), flush=True)
print("row1 idx[-1] =", idx2[1, 0, -1].item(), flush=True)
print("  -> row0 should be 1, row1 should be 2", flush=True)
