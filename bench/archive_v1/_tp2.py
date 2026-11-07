import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.tp import Shard, shard_linear, asplit, shard_forward
import torch.nn as nn

print('RESULT === asplit ===')
for n, caps in ((384, [1, 1]), (384, [1, 1, 1, 1]), (100, [3, 1]), (7, [1, 1])):
    sp = asplit(n, caps)
    tot = sum(hi - lo for lo, hi in sp)
    cover = all(lo <= hi for lo, hi in sp)
    print(f'  n={n} caps={caps}: {sp} sum={tot} ok={tot==n}')

print('RESULT === shard_linear (out axis) ===')
torch.manual_seed(0)
lin = nn.Linear(64, 128, bias=False).cuda()
x = torch.randn(4, 64, device='cuda')
full = lin(x)
# 分成 2 片
shards = []
for lo, hi in ((0, 64), (64, 128)):
    s = Shard(axis='out', lo=lo, hi=hi, size=128)
    shards.append(shard_linear(lin, s, axis='out')(x))
cat = torch.cat(shards, dim=-1)
print(f'  out-axis: full{tuple(full.shape)} sharded{tuple(cat.shape)} maxdiff={(full-cat).abs().max().item():.3e}')

print('RESULT === shard_linear (in axis) ===')
lin2 = nn.Linear(64, 32, bias=False).cuda()
x2 = torch.randn(4, 64, device='cuda')
full2 = lin2(x2)
acc = None
for lo, hi in ((0, 32), (32, 64)):
    s = Shard(axis='in', lo=lo, hi=hi, size=64)
    part = shard_linear(lin2, s, axis='in')(x2[..., lo:hi])
    acc = part if acc is None else acc + part
print(f'  in-axis: maxdiff={(full2-acc).abs().max().item():.3e}')