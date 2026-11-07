
import torch, time
import tinyinfer._C_stable_libtorch  # noqa
from tinyinfer import moe_gemv

dev = "cuda"
E, N, K = 6, 2048, 7168
padded_k = ((K // 32) + 3) // 4

torch.manual_seed(0)

def make_w(n, k):
    w = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device=dev)
    s = torch.full((n, k // 32), 126, dtype=torch.uint8, device=dev)
    return w, s

w1s, s1s, w3s, s3s = [], [], [], []
for _ in range(E):
    w1, s1 = make_w(N, K); w1s.append(w1); s1s.append(s1)
    w3, s3 = make_w(N, K); w3s.append(w3); s3s.append(s3)
w2s, s2s = [], []
for _ in range(E):
    w2, s2 = make_w(K, N); w2s.append(w2); s2s.append(s2)

x = torch.randn(1, K, dtype=torch.bfloat16, device=dev)
aq, asc = None, None
try:
    from tinyinfer.gemv_fp4 import quantize_act
    aq, asc = quantize_act(x, K)
except Exception as e:
    print("quantize_act failed:", e); raise SystemExit(0)

# ---- moe_gemv.gate_up ----
gu = moe_gemv.gate_up(aq, asc, w1s, s1s, w3s, s3s)
torch.cuda.synchronize()
print("moe_gemv gate_up out:", tuple(gu.shape), gu.dtype)
for _ in range(5): moe_gemv.gate_up(aq, asc, w1s, s1s, w3s, s3s)
torch.cuda.synchronize()
t0 = time.perf_counter(); R = 30
for _ in range(R): moe_gemv.gate_up(aq, asc, w1s, s1s, w3s, s3s)
torch.cuda.synchronize()
mv_us = (time.perf_counter() - t0) / R * 1e6
print(f"moe_gemv.gate_up: {mv_us:.1f} us")

# ---- grouped: 堆叠 w1|w3 成 [E, 2N, K/2] ----
t0 = time.perf_counter()
w_gu = torch.stack([torch.cat([w1s[e], w3s[e]], 0) for e in range(E)], 0)
s_gu = torch.stack([torch.cat([s1s[e], s3s[e]], 0) for e in range(E)], 0)
torch.cuda.synchronize()
stack_us = (time.perf_counter() - t0) * 1e6

# 激活量化到 grouped 需要的形式
M = 1
a_q = torch.empty(E * M, K // 2, dtype=torch.uint8, device=dev)
a_sf = torch.empty(256, padded_k, dtype=torch.int32, device=dev)
eo = torch.arange(0, E + 1, dtype=torch.int32, device=dev)
x_rep = x.repeat(E, 1)
torch.ops._C.mxfp4_experts_quant(a_q, a_sf, x_rep, eo, eo, E)
a_sf8 = a_sf.view(torch.uint8)[: E * M]
s_gu3 = s_gu.view(torch.uint8).view(E, 2 * N, padded_k * 4)
out = torch.empty(E * M, 2 * N, dtype=torch.bfloat16, device=dev)
ps = torch.tensor([[M, 2 * N, K]] * E, dtype=torch.int32, device=dev)
eo_lo = eo[:-1].contiguous()
so_lo = eo[:-1].contiguous()

torch.ops._C.cutlass_mxfp4_group_mm(out, a_q, w_gu, a_sf8, s_gu3, ps, eo_lo, so_lo)
torch.cuda.synchronize()
for _ in range(5): torch.ops._C.cutlass_mxfp4_group_mm(out, a_q, w_gu, a_sf8, s_gu3, ps, eo_lo, so_lo)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(R): torch.ops._C.cutlass_mxfp4_group_mm(out, a_q, w_gu, a_sf8, s_gu3, ps, eo_lo, so_lo)
torch.cuda.synchronize()
gr_us = (time.perf_counter() - t0) / R * 1e6
print(f"grouped gate_up : {gr_us:.1f} us  (stack cost one-off {stack_us:.0f} us)")
print(f"  ratio grouped/moe_gemv = {gr_us/mv_us:.2f}x")
