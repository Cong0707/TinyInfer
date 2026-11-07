"""Correctness with the *same* byte semantics the kernel uses: uint32 words."""
import sys
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
import torch
import tinyinfer.torch_compat  # noqa
from tinyinfer import native

N, K = 256, 512
# build words from uint32 domain explicitly
w_np = torch.randint(0, 2**31 - 1, (N, K//8), dtype=torch.int64)
w = w_np.to(torch.int32).to("cuda")
ws = torch.full((N, K//32), 127, dtype=torch.uint8, device="cuda")
x = torch.randn(K, dtype=torch.bfloat16, device="cuda")

# reference: interpret each word as UNSIGNED 32-bit, take nibbles low->high
u = w_np & 0xFFFFFFFF
codes = torch.zeros((N, K), dtype=torch.int64)
for i in range(8):
    codes[:, i::8] = (u >> (4 * i)) & 0xF
signed = codes.clone()
signed[signed >= 8] -= 16
ref = (signed.float() @ x.cpu().float())

y = torch.zeros(N, dtype=torch.bfloat16, device="cuda")
native.launch(native.kernel("marlin_fp4","marlin_gemv2"), ((N + 63)//64,), (128,),
              [x, w, ws, y, N, K])
torch.cuda.synchronize()
d = (y.cpu().float() - ref).abs()
print(f"max {d.max().item():.4f}  mean {d.mean().item():.4f}")
print(f"  y  [0:6]: {y[:6].tolist()}")
print(f"  ref[0:6]: {ref[:6].tolist()}")
print(f"  rel: {(d.max()/ref.abs().max()).item():.2e}")