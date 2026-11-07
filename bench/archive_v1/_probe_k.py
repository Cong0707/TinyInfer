"""Which k does the kernel actually multiply against? Put a 1 at one k and read the output."""
import sys
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
import torch
import tinyinfer.torch_compat  # noqa: F401
from tinyinfer import native

N, K = 64, 32
# every code = 1 (nibble value 1) so each weight contributes exactly its code (1-8 = -7)
w = torch.zeros((N, K//8), dtype=torch.int32, device="cuda")
w[:, :] = 0x11111111
ws = torch.full((N, K//32), 127, dtype=torch.uint8, device="cuda")

# activation: single 1 at position j, rest 0 -> output = weight[k=j]
for j in (0, 1, 2, 4, 8):
    x = torch.zeros(K, dtype=torch.bfloat16, device="cuda")
    x[j] = 1.0
    y = torch.zeros(N, dtype=torch.bfloat16, device="cuda")
    native.launch(native.kernel("marlin_fp4","marlin_gemv2"), ((N + 63)//64,), (128,),
                  [x, w, ws, y, N, K])
    torch.cuda.synchronize()
    print(f"act[0]={j:2d} -> y[0] = {y[0].item():8.2f}  (weight code at k={j} maps to this)")

# also check: is the output the same for all rows? (16 rows share the tile)
x = torch.zeros(K, dtype=torch.bfloat16, device="cuda"); x[0] = 1.0
y = torch.zeros(N, dtype=torch.bfloat16, device="cuda")
native.launch(native.kernel("marlin_fp4","marlin_gemv2"), ((N + 63)//64,), (128,), [x, w, ws, y, N, K])
torch.cuda.synchronize()
print()
print("rows 0..15 with act[0]=1:", y[:16].tolist())
