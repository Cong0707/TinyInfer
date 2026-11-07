
import torch
import torch.nn.functional as F
from tinyinfer.vllm_ops import clamp_swiglu
import tinyinfer.vllm_ops as vo

print("ops loaded:", vo.available())
print("moe ops loaded:", vo._moe_ops() is not None)

limit = 8.0
x = torch.randn(4, 16384, dtype=torch.bfloat16, device="cuda") * 4.0
out = clamp_swiglu(x, limit)
print("fused:", out is not None, "shape:", None if out is None else tuple(out.shape))

if out is not None:
    half = x.shape[-1] // 2
    gate, up = x[..., :half].float(), x[..., half:].float()
    ref = F.silu(torch.clamp(gate, max=limit)) * torch.clamp(up, min=-limit, max=limit)
    diff = (out.float() - ref).abs()
    print("max abs diff:", float(diff.max()), " max rel:", float((diff / (ref.abs() + 1e-3)).max()))
    print("allclose bf16 tol:", torch.allclose(out, ref.to(torch.bfloat16), atol=2e-2, rtol=2e-2))

# 性能
import time
torch.cuda.synchronize()
for _ in range(20):
    clamp_swiglu(x, limit)
torch.cuda.synchronize()
t0 = time.perf_counter()
N = 300
for _ in range(N):
    clamp_swiglu(x, limit)
torch.cuda.synchronize()
fused_us = (time.perf_counter() - t0) / N * 1e6

gate, up = x[..., :half], x[..., half:]
for _ in range(20):
    u = torch.clamp(up.float(), min=-limit, max=limit)
    g = torch.clamp(gate.float(), max=limit)
    _ = F.silu(g) * u
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(N):
    u = torch.clamp(up.float(), min=-limit, max=limit)
    g = torch.clamp(gate.float(), max=limit)
    _ = F.silu(g) * u
torch.cuda.synchronize()
ref_us = (time.perf_counter() - t0) / N * 1e6
print(f"fused {fused_us:.2f}us  torch {ref_us:.2f}us  speedup {ref_us/fused_us:.2f}x")
