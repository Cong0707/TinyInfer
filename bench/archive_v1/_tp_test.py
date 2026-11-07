
import torch
from tinyinfer.tp import shard_forward, asplit, Pool, Device

# 单卡上用伪造的 plan 验证：把一行拆成两段由同一张卡算，结果必须和整体算一致
N, K = 2304, 512
w = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
x = torch.randn(1, K, dtype=torch.bfloat16, device="cuda")
ref = x @ w.T

def fn(dev, lo, hi, xs):
    return xs @ w[lo:hi].T

plan = [(0, (0, 1280)), (0, (1280, N))]
out = shard_forward(x, plan, fn)
print("2-range same card : maxdiff", (out.float() - ref.float()).abs().max().item())

pool = Pool([Device(0, "fake", 1 << 40, 1 << 40, 1.0)])
plan1 = [(0, (0, N))]
out1 = shard_forward(x, plan1, fn)
print("1-range           : maxdiff", (out1.float() - ref.float()).abs().max().item())

# asplit 的边界必须 128 对齐，且覆盖完整
for caps in ([3, 1], [95, 31], [1, 1], [7]):
    rs = asplit(N, caps)
    ok = rs[0][0] == 0 and rs[-1][1] == N
    aligned = all(lo % 128 == 0 and hi % 128 == 0 for lo, hi in rs)
    cont = all(rs[i][1] == rs[i + 1][0] for i in range(len(rs) - 1))
    print(f"asplit({N},{caps}) -> {rs}  cover={ok} align={aligned} contiguous={cont}")
