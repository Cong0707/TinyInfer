import sys
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
import torch
import tinyinfer.torch_compat  # noqa
from tinyinfer import native

N, K = 256, 512
word = sum((i & 0xF) << (4*i) for i in range(8))
w = torch.full((N, K//8), word, dtype=torch.int32, device="cuda")
ws = torch.full((N, K//32), 127, dtype=torch.uint8, device="cuda")
torch.manual_seed(0)
x = torch.randn(K, dtype=torch.bfloat16, device="cuda")

y = torch.zeros(N, dtype=torch.bfloat16, device="cuda")
native.launch(native.kernel("marlin_fp4","marlin_gemv2"), ((N + 63)//64,), (128,), [x, w, ws, y, N, K])
torch.cuda.synchronize()
codes = []
for j in range(K):
    c = (j % 8) & 0xF
    codes.append(c - 16 if c >= 8 else c)
ref = sum(c * x[j].item() for j, c in enumerate(codes))
print(f"distinct: {len(torch.unique(y))} (expect 1)")
print(f"y[0] = {y[0].item():.4f}")
print(f"ref  = {ref:.4f}")
print(f"diff = {abs(y[0].item()-ref):.4f}")
