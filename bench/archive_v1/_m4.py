import sys
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
import torch
import tinyinfer.torch_compat  # noqa
from tinyinfer import native

N, K = 256, 512
# every word = same pattern, so no signed/unsigned ambiguity at all
word = sum((i & 0xF) << (4*i) for i in range(8))   # nibbles 0..7
w = torch.full((N, K//8), word, dtype=torch.int32, device="cuda")
ws = torch.full((N, K//32), 127, dtype=torch.uint8, device="cuda")
torch.manual_seed(0)
x = torch.randn(K, dtype=torch.bfloat16, device="cuda")

# every row has identical weights -> every output must be identical
y = torch.zeros(N, dtype=torch.bfloat16, device="cuda")
native.launch(native.kernel("marlin_fp4","marlin_gemv2"), ((N + 63)//64,), (128,),
              [x, w, ws, y, N, K])
torch.cuda.synchronize()
uniq = torch.unique(y)
print(f"distinct outputs: {len(uniq)} (expect 1)")
print(f"y[0]={y[0].item():.4f}  y[64]={y[64].item():.4f}  y[128]={y[128].item():.4f}")

codes = []
for j in range(K):
    c = (j % 8) & 0xF
    codes.append(c - 16 if c >= 8 else c)
ref = sum(c * x[j].item() for j, c in enumerate(codes))
print(f"reference: {ref:.4f}")
print(f"diff: {abs(y[0].item() - ref):.4f}")