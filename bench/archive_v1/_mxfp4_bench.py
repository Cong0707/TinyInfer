
import torch, time
import tinyinfer._C_stable_libtorch  # noqa

dev = "cuda"
E, N, K = 6, 2048, 7168
m_per = 1
M = E * m_per
MAX_TOK = 128
padded_k = ((K // 32) + 3) // 4

torch.manual_seed(0)
x = torch.randn(M, K, dtype=torch.bfloat16, device=dev)
w = torch.randn(E * N, K, dtype=torch.bfloat16, device=dev)

expert_offsets = torch.arange(0, M + 1, m_per, dtype=torch.int32, device=dev)
bs_offsets = torch.arange(0, E + 1, dtype=torch.int32, device=dev) * m_per

def prep(t, eoff, boff, rows, max_tok):
    q = torch.empty(rows, K // 2, dtype=torch.uint8, device=dev)
    sf = torch.empty(max_tok, padded_k, dtype=torch.int32, device=dev)
    torch.ops._C.mxfp4_experts_quant(q, sf, t, eoff, boff, len(eoff) - 1)
    return q, sf.view(torch.uint8)

a_q, a_sf = prep(x, expert_offsets, bs_offsets, M, 128)
w_q, w_sf = prep(w, torch.tensor([g * N for g in range(E)] + [E * N], dtype=torch.int32, device=dev),
                 torch.tensor([g * N for g in range(E)] + [E * N], dtype=torch.int32, device=dev), E * N, E * N)
w_q3 = w_q.view(E, N, K // 2)
w_sf3 = w_sf[: E * N].view(E, N, padded_k * 4)
a_sf2 = a_sf[:M]

out = torch.empty(M, N, dtype=torch.bfloat16, device=dev)
ps = torch.tensor([[m_per, N, K]] * E, dtype=torch.int32, device=dev)
eo = expert_offsets[:-1].contiguous()
so = bs_offsets[:-1].contiguous()

torch.ops._C.cutlass_mxfp4_group_mm(out, a_q, w_q3, a_sf2, w_sf3, ps, eo, so)
torch.cuda.synchronize()

# 计时 GEMM 单次
for _ in range(10):
    torch.ops._C.cutlass_mxfp4_group_mm(out, a_q, w_q3, a_sf2, w_sf3, ps, eo, so)
torch.cuda.synchronize()
t0 = time.perf_counter()
REP = 50
for _ in range(REP):
    torch.ops._C.cutlass_mxfp4_group_mm(out, a_q, w_q3, a_sf2, w_sf3, ps, eo, so)
torch.cuda.synchronize()
gemm_us = (time.perf_counter() - t0) / REP * 1e6

weight_bytes = E * N * K / 2          # fp4 packed
print(f"E={E} N={N} K={K}  M={M}")
print(f"  weights: {weight_bytes/1e6:.1f} MB")
print(f"  GEMM one shot: {gemm_us:.1f} us  -> {weight_bytes/gemm_us/1e3:.1f} GB/s")
print(f"  out[0,:3]: {[round(v,3) for v in out[0,:3].tolist()]}")
