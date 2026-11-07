import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.hc import hc_split_sinkhorn

torch.manual_seed(0)
b, s, hc = 2, 8, 4
n = hc + hc + hc * hc  # pre + post + comb
mixes = torch.randn(b, s, n)
scale = torch.rand(3).abs() + 0.5
base = torch.randn(n) * 0.1

_, _, comb20 = hc_split_sinkhorn(mixes, scale, base, hc, 20)
_, _, comb6 = hc_split_sinkhorn(mixes, scale, base, hc, 6)
_, _, comb3 = hc_split_sinkhorn(mixes, scale, base, hc, 3)
print("comb 20 vs 6 max abs diff:", (comb20 - comb6).abs().max().item())
print("comb 20 vs 3 max abs diff:", (comb20 - comb3).abs().max().item())
print("max comb magnitude:", comb20.abs().max().item())
print("row sums (20):", comb20.sum(-1).flatten()[:4].tolist())
