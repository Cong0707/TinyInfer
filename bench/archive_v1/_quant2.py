
import torch
import tinyinfer._C_stable_libtorch  # noqa

dev = "cuda"
E, N, K = 4, 512, 512
m_per = 8
M = E * m_per
MAX_TOK = 128
scales_k = K // 32
padded_k = (scales_k + 3) // 4

x = torch.randn(M, K, dtype=torch.bfloat16, device=dev)
expert_offsets = torch.arange(0, M + 1, m_per, dtype=torch.int32, device=dev)
bs_offsets = (torch.arange(E + 1, dtype=torch.int32, device=dev) * m_per)

out = torch.empty(M, K // 2, dtype=torch.uint8, device=dev)
out_sf = torch.empty(MAX_TOK, padded_k, dtype=torch.int32, device=dev)
try:
    torch.ops._C.mxfp4_experts_quant(out, out_sf, x, expert_offsets, bs_offsets, E)
    torch.cuda.synchronize()
    sf8 = out_sf.view(torch.uint8)
    print("quant OK  out:", tuple(out.shape), " sf8:", tuple(sf8.shape), "numel", sf8.numel())
    a_bs = sf8[:M]           # 2D: [rows, padded_k*4]
    print("a_bs slice:", tuple(a_bs.shape), " need k//32 =", scales_k)
except Exception as exc:
    print("quant FAILED:", type(exc).__name__, str(exc)[:300])
