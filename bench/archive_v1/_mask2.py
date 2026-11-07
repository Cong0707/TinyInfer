
import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.nn import window_idxs_per_row

W, old_len = 4, 0
# 语义：positions = 每行「这一列」的绝对位置
# 行0 这一列在位置 0（1 token 的 prompt），行1 这一列在位置 3
# 行0 只能看位置 0；行1 能看 0,1,2,3
pos = torch.tensor([0, 3])
got = window_idxs_per_row(pos, 1, old_len, W)
print("per_row:", got.tolist())
print("期望: 行0=[-1,-1,-1,0]  行1=[0,1,2,3]")
ok = got[0].tolist() == [[-1,-1,-1,0]] and got[1].tolist() == [[0,1,2,3]]
print("正确:", ok)
