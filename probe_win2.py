"""window 索引在不同长度下的尾部。"""
import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.nn import window_idxs_ring
for N in (8, 11, 12, 13):
    idx = window_idxs_ring(0, N, 0, 128, "cpu")
    print("N=%2d shape=%s" % (N, tuple(idx.shape)))
    for p in range(max(0, N - 3), N):
        row = idx[p]
        valid = row[row >= 0]
        neg = int((row < 0).sum())
        print("   pos %2d: n_valid=%3d  first=%s  last=%s  max=%d" % (
              p, len(valid), valid[:3].tolist(), valid[-3:].tolist(), int(row.max())))
