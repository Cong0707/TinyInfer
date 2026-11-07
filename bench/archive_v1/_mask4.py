
import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.nn import window_idxs_per_row
W = 4
pos = torch.tensor([0, 4]); old = [0, 4]
g = window_idxs_per_row(pos, 1, old, W)
print("per-row old_len:", g.tolist())
print("期望 行0=[-1,-1,-1,0] 行1=[1,2,3,4] =>", g[0].tolist()==[[-1,-1,-1,0]] and g[1].tolist()==[[1,2,3,4]])
