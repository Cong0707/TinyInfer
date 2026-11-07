
import torch
import tinyinfer._C_stable_libtorch  # noqa

E, N, K = 4, 512, 512
m_per = 8
M = E * m_per
dev = "cuda"

x = torch.randn(M, K, dtype=torch.bfloat16, device=dev)
expert_offsets = torch.arange(0, M + 1, m_per, dtype=torch.int32, device=dev)
bs_offsets = (torch.arange(E + 1, dtype=torch.int32, device=dev) * m_per)

try:
    q, sf = torch.ops._C.mxfp4_experts_quant(x, expert_offsets, bs_offsets, E, 1)
    print("mxfp4_experts_quant OK")
    print("  q :", q.shape, q.dtype)
    print("  sf:", sf.shape, sf.dtype, "numel", sf.numel())
    scales_per_row = K // 32
    print("  expected sf flat:", E * m_per * scales_per_row)
except Exception as exc:
    print("mxfp4_experts_quant FAILED:", type(exc).__name__, str(exc)[:300])
