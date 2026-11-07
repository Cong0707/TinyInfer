"""Correctness first: mma path vs an int4 numpy reference, with scales."""
import sys
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
import torch
import tinyinfer.torch_compat  # noqa: F401
from tinyinfer import native

N, K = 256, 512
w = torch.randint(0, 2**31, (N, K//8), dtype=torch.int32, device="cuda")
ws = torch.full((N, K//32), 127, dtype=torch.uint8, device="cuda")  # e8m0 = 2^0
x = torch.randn(K, dtype=torch.bfloat16, device="cuda")

codes = torch.zeros((N, K), dtype=torch.int64, device="cuda")
u = w.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
for i in range(8):
    codes[:, i::8] = (u >> (4 * i)) & 0xF
signed = codes.clone()
signed[signed >= 8] -= 16
ref = (signed.float() @ x.float())

y = torch.zeros(N, dtype=torch.bfloat16, device="cuda")
native.launch(native.kernel("marlin_fp4","marlin_gemv2"), ((N + 63)//64,), (128,),
              [x, w, ws, y, N, K])
torch.cuda.synchronize()
d = (y.float() - ref).abs()
print(f"max {d.max().item():.4f}  mean {d.mean().item():.4f}")
print(f"  y  [0:6]: {y[:6].tolist()}")
print(f"  ref[0:6]: {ref[:6].tolist()}")
print(f"  rel: {(d.max()/ref.abs().max()).item():.2e}")
