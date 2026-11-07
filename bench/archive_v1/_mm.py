
import torch
torch.manual_seed(0)
K, N = 5120, 2304
W = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
x = torch.randn(1, K, dtype=torch.bfloat16, device="cuda")

# 同样一行，一次算 1 行，一次算 2 行（两行内容相同）
y1 = x @ W.T
x2 = x.repeat(2, 1).contiguous()
y2 = x2 @ W.T
print("matmul M=1 vs M=2[0] maxdiff:", (y1[0] - y2[0]).abs().max().item())

# fp8 路径
from tinyinfer.gemm import fp8_linear
w8 = W.to(torch.float8_e4m3fn)
n32, k32 = N // 32, K // 32
s8 = torch.ones(n32, k32, dtype=torch.float8_e8m0fnu, device="cuda")
a1 = fp8_linear(x, w8, s8)
a2 = fp8_linear(x2, w8, s8)
print("fp8_linear M=1 vs M=2[0] maxdiff:", (a1[0] - a2[0]).abs().max().item())
