import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.nn import window_idxs_per_row
# 两行：pos0=0（新行）、pos1=4（已走 4 步），window=128
pos = torch.tensor([0, 4], device="cuda")
old = [0, 4]
idx = window_idxs_per_row(pos, 2, old, 128, "cuda")
print("shape", tuple(idx.shape))
print("row0", idx[0].tolist())
print("row1", idx[1].tolist())
print("min", int(idx.min()), "max", int(idx.max()))
