
import sys
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
import torch
# 算多序列 KV 需求
from tinyinfer.config import TinyConfig
c = TinyConfig.from_hf(r"E:\models\DeepSeek-V4.1-Flash-1017")
win = c.sliding_window           # 128
hd = c.head_dim                  # 512
nl = c.n_layers                  # 40
bytes_per_seq_win = win * hd * 2 * nl
print(f"窗口 KV/序列: {bytes_per_seq_win/2**20:.1f} MB  ({win}x{hd}x2B x {nl} 层)")
# 压缩 KV：每 ratio token 一组
ratios = c.compress_ratios
for ctx in (1024, 4096):
    tot = 0
    for r in ratios:
        if r > 0:
            groups = ctx // r
            tot += groups * hd * 2
    print(f"压缩 KV/序列 @{ctx} tokens: {tot/2**20:.1f} MB")
print()
print("8 序列 @4K 上下文约需:", round((bytes_per_seq_win*8 + 8*sum((4096//r)*hd*2 for r in ratios if r>0))/2**20), "MB")
print("当前可用显存: 约 6900 MB，其中模型常驻已占大部分")
