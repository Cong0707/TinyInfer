"""Is the mma path numerically right? Compare against a numpy reference."""
import sys
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
import torch
import tinyinfer.torch_compat  # noqa: F401
from tinyinfer import native

N, K = 256, 512
w = torch.randint(0, 2**31, (N, K//8), dtype=torch.int32, device="cuda")
x = torch.randn(K, dtype=torch.bfloat16, device="cuda")

# decode reference in torch: 8 nibbles per uint32, first nibble = lowest 4 bits
wc = w.view(torch.uint8) if hasattr(w, "view") else w.to(torch.int32).view(torch.uint8)
# packed nibbles: get 4-bit codes
codes = torch.zeros((N, K), dtype=torch.int64, device="cuda")
u = w.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
for i in range(8):
    codes[:, i::8] = (u >> (4 * i)) & 0xF

# E2M1 value table (what deq4 produces, mapped from int4 -8..7 back to E2M1)
# deq4 gives (code_as_int4)  ->  the nibble interpreted as signed 4-bit
signed = codes.clone()
signed[signed >= 8] -= 16
# our kernel's w is already the fp4 code; the reference must use the same mapping
# deq4 produces the signed-int4 interpretation directly (no E2M1 table), so:
ref = torch.zeros(N, dtype=torch.float32, device="cuda")
xf = x.float()
for r in range(0, N, 64):
    pass
# do it properly: sum over k of signed[r,k] * x[k]
ref = (signed.float() @ xf).float()

y = torch.zeros(N, dtype=torch.bfloat16, device="cuda")
native.launch(native.kernel("marlin_fp4","marlin_gemv"), ((N + 63)//64,), (128,),
              [x, w, y, N, K])
torch.cuda.synchronize()
diff = (y.float() - ref).abs()
print(f"mma vs int4 reference: max {diff.max().item():.4f}, mean {diff.mean().item():.4f}")
print(f"  y[0:4]  : {y[:4].tolist()}")
print(f"  ref[0:4]: {ref[:4].tolist()}")
print(f"  rel err: {(diff.max()/ref.abs().max()).item():.2e}")
