"""Minimal case: K=32 (one block), N=16 (one warp). Verify deq8 order in-kernel."""
import sys
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
import torch
import tinyinfer.torch_compat  # noqa: F401
from tinyinfer import native

N, K = 64, 32
# one word = codes 0..7 laid out as nibbles 0..7
w0 = 0
for i in range(8):
    w0 |= (i & 0xF) << (4 * i)
w = torch.zeros((N, K//8), dtype=torch.int32, device="cuda")
w[:, 0] = w0; w[:, 1] = w0; w[:, 2] = w0; w[:, 3] = w0
ws = torch.full((N, K//32), 127, dtype=torch.uint8, device="cuda")
x = torch.ones(K, dtype=torch.bfloat16, device="cuda")

# reference: codes 0..7 per word, 4 words = 32 k, each row identical
codes = []
for word in range(4):
    for i in range(8):
        codes.append(i)
ref_val = sum(c - (16 if c >= 8 else 0) for c in codes)
ref = torch.full((N,), float(ref_val), device="cuda")
print(f"expected each row = {ref_val} (sum of codes 0..7 repeated 4x)")

y = torch.zeros(N, dtype=torch.bfloat16, device="cuda")
native.launch(native.kernel("marlin_fp4","marlin_gemv2"), ((N + 63)//64,), (128,),
              [x, w, ws, y, N, K])
torch.cuda.synchronize()
print("y:", y[:8].tolist())
print("all equal:", bool((y == y[0]).all()))
