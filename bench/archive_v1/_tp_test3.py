
import torch
from tinyinfer.tp import asplit, shard_quant_linear
from tinyinfer.linear import QuantLinear

N, K = 2304, 5120
# fp8 权重 + [N/32, K/32] 的块 scale
w = torch.randn(N, K, dtype=torch.bfloat16, device="cuda").to(torch.float8_e4m3fn)
s = torch.ones(N // 32, K // 32, dtype=torch.float8_e8m0fnu, device="cuda")
x = torch.randn(1, K, dtype=torch.bfloat16, device="cuda")

full = QuantLinear(w, s)
ref = full.forward(x)
print("ref:", tuple(ref.shape), " out_features:", full.out_features, " sharded:", full.is_sharded)

for caps in ([1, 1], [3, 1], [1, 1, 1]):
    plan = [(0, r) for r in asplit(N, caps)]
    sl = shard_quant_linear(w, s, plan)
    y = sl.forward(x)
    d = (y.float() - ref.float()).abs().max().item()
    print(f"caps={caps} ranges={[r for _, r in plan]}  maxdiff={d:.3e}  "
          f"shape={tuple(y.shape)} out_features={sl.out_features} sharded={sl.is_sharded}")

# 单段退化：必须和原来逐位相同
sl1 = shard_quant_linear(w, s, [(0, (0, N))])
y1 = sl1.forward(x)
print("single-range maxdiff:", (y1.float()-ref.float()).abs().max().item(), " sharded:", sl1.is_sharded)
