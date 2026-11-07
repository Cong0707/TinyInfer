
import torch
import tinyinfer._C_stable_libtorch  # noqa

dev = "cuda"
E, N, K = 4, 512, 512
m_per = 8
M = E * m_per
MAX_TOK = 128
scales_k = K // 32
padded_k = (scales_k + 3) // 4

torch.manual_seed(0)
x = torch.randn(M, K, dtype=torch.bfloat16, device=dev)
expert_offsets = torch.arange(0, M + 1, m_per, dtype=torch.int32, device=dev)
bs_offsets = (torch.arange(E + 1, dtype=torch.int32, device=dev) * m_per)

a_q = torch.empty(M, K // 2, dtype=torch.uint8, device=dev)
a_sf = torch.empty(MAX_TOK, padded_k, dtype=torch.int32, device=dev)
try:
    torch.ops._C.mxfp4_experts_quant(a_q, a_sf, x, expert_offsets, bs_offsets, E)
    torch.cuda.synchronize()
    print("quant OK  a_q:", tuple(a_q.shape), " a_sf8:", tuple(a_sf.view(torch.uint8).shape))
except Exception as exc:
    print("quant FAILED:", type(exc).__name__, str(exc)[:260])
    raise SystemExit(0)

# 权重也用同一个 kernel 量化（每个专家 N 行当 tokens）
w = torch.randn(E * N, K, dtype=torch.bfloat16, device=dev)
w_expert_off = torch.tensor([g * N for g in range(E)] + [E * N], dtype=torch.int32, device=dev)
w_bs_off = torch.tensor([g * N for g in range(E)] + [E * N], dtype=torch.int32, device=dev)
w_q = torch.empty(E * N, K // 2, dtype=torch.uint8, device=dev)
w_sf = torch.empty(E * N, padded_k, dtype=torch.int32, device=dev)
torch.ops._C.mxfp4_experts_quant(w_q, w_sf, w, w_expert_off, w_bs_off, E)
torch.cuda.synchronize()
w_q3 = w_q.view(E, N, K // 2)
w_sf3 = w_sf.view(torch.uint8)[: E * N].view(E, N, padded_k * 4)
print("weight quant OK:", tuple(w_q3.shape), tuple(w_sf3.shape))

out = torch.empty(M, N, dtype=torch.bfloat16, device=dev)
problem_sizes = torch.tensor([[m_per, N, K]] * E, dtype=torch.int32, device=dev)
try:
    torch.ops._C.cutlass_mxfp4_group_mm(
        out, a_q, w_q3, a_sf.view(torch.uint8)[:M], w_sf3, problem_sizes,
        expert_offsets[:-1].contiguous(), bs_offsets[:-1].contiguous())
    torch.cuda.synchronize()
    ref = torch.zeros(M, N, dtype=torch.float32, device=dev)
    for g in range(E):
        ref[g * m_per:(g + 1) * m_per] = x[g * m_per:(g + 1) * m_per].float() @ w[g * N:(g + 1) * N].float().T
    def calc_diff(x, y):
        x, y = x.double(), y.double()
        denom = (x * x + y * y).sum()
        return float(1 - 2 * (x * y).sum() / denom)

    diffs = []
    for g in range(E):
        sl = slice(g * m_per, (g + 1) * m_per)
        diffs.append(calc_diff(out[sl].float(), ref[sl]))
    print("GEMM OK  per-expert cosine diff:", [round(d, 5) for d in diffs])
    print("  max diff:", round(max(diffs), 5), " (official threshold 0.15)")
except Exception as exc:
    print("GEMM FAILED:", type(exc).__name__, str(exc)[:600])
