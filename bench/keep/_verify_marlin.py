
import sys
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
import torch
from tinyinfer.gemv_fp4_marlin import gemv_fp4_fast

dev = torch.device("cuda")
N, K = 256, 512
torch.manual_seed(0)
w_fp4 = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=dev)
w_scale = torch.randint(120, 130, (N, K // 32), dtype=torch.uint8, device=dev)
x = torch.randn(1, K, dtype=torch.float32, device=dev)

y = gemv_fp4_fast(x, w_fp4, w_scale).float().reshape(-1)

# reference: decode E2M1 by hand.
# low nibble is the even index, high nibble the odd one (vLLM's packing order).
LUT = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], device=dev)
lo = w_fp4 & 0x0F
hi = (w_fp4 >> 4) & 0x0F
w = torch.stack([LUT[lo.long()], LUT[hi.long()]], dim=-1).reshape(N, K)
e8m0 = w_scale.to(torch.int32) - 127
s = torch.pow(2.0, e8m0.float()).repeat_interleave(32, dim=1)[:, :K]
w = w * s
ref = (w @ x.reshape(-1)).reshape(-1)

err = (y - ref).abs().max().item()
rel = err / ref.abs().max().item()
print(f"max abs err {err:.5f}   rel {rel:.3e}   ref scale {ref.abs().max().item():.3f}")
print("y[:4]  ", y[:4].tolist())
print("ref[:4]", ref[:4].tolist())
print("PASS" if rel < 1e-2 else "MISMATCH")
