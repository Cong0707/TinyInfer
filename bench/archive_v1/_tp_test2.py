
import torch
from tinyinfer.tp import ShardedLinear, asplit

class Lin:
    """stand-in for QuantLinear: holds rows [lo,hi) of a single matrix."""
    def __init__(self, w):
        self.weight = w
        self.out_features = w.shape[0]
    def to(self, d):
        self.weight = self.weight.to(d); return self
    def forward(self, x):
        return x @ self.weight.T

N, K = 2304, 512
w = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
x = torch.randn(1, K, dtype=torch.bfloat16, device="cuda")
ref = x @ w.T

# 同一张卡上分两段，必须和整体逐位相同
parts = [(0, Lin(w[0:1792].clone())), (0, Lin(w[1792:].clone()))]
sl = ShardedLinear(parts)
out = sl(x)
print("sharded(1792+512)  maxdiff:", (out.float()-ref.float()).abs().max().item(), " shape:", tuple(out.shape))
print("out_features:", sl.out_features, " ref:", N)

# 单段退化
sl1 = ShardedLinear([(0, Lin(w.clone()))])
o1 = sl1(x)
print("single-part        maxdiff:", (o1.float()-ref.float()).abs().max().item())

# 三段，验证拼接顺序
rs = asplit(N, [1, 1, 1])
pp = [(0, Lin(w[lo:hi].clone())) for lo, hi in rs]
o3 = ShardedLinear(pp)(x)
print("asplit", rs, "maxdiff:", (o3.float()-ref.float()).abs().max().item())
