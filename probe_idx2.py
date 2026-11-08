import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.nn import window_idxs_ring_cached, window_idxs_per_row

W = 128
# 单行路径（pos=1, s=1, old=1）
a = window_idxs_ring_cached(1, 1, W, "cpu")
print("single: nonzero=%d idx[-1]=%d" % (int((a>=0).sum()), a[0,-1].item()), flush=True)

# per-row 路径（pos=[1,1], old=[1,1]）
b = window_idxs_per_row(torch.tensor([1,1]), 1, [1,1], W, "cpu")
print("perrow: row0 nonzero=%d idx[-1]=%d" % (int((b[0]>=0).sum()), b[0,0,-1].item()), flush=True)
print("perrow row0:", b[0,0].tolist()[-4:], flush=True)
print("single     :", a[0].tolist()[-4:], flush=True)
