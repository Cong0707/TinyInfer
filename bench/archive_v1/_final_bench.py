
import time, torch
import tinyinfer._C_stable_libtorch  # noqa
import tinyinfer._moe_C_stable_libtorch  # noqa

dev = "cuda"
def bench(fn, R=50, w=20):
    for _ in range(w): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(R): fn()
    torch.cuda.synchronize()
    return (time.perf_counter()-t0)/R*1e6

print("cap:", torch.cuda.get_device_capability())
ops = [n for n in torch._C._dispatch_get_all_op_names() if n.startswith("_C::") or n.startswith("_moe_C::")]
print(f"ops: {len(ops)} (_C {len([o for o in ops if o.startswith('_C::')])} / _moe_C {len([o for o in ops if o.startswith('_moe_C::')])})")

# --- mxfp4 grouped MoE 复测 ---
E, N, K = 6, 2048, 7168
padded_k = ((K // 32) + 3) // 4
w_gu = torch.randint(0, 256, (E, 2*N, K//2), dtype=torch.uint8, device=dev)
s_gu = torch.full((E, 2*N, K//32), 126, dtype=torch.uint8, device=dev)
x = torch.randn(1, K, dtype=torch.bfloat16, device=dev)
M = 1
a_q = torch.empty(E*M, K//2, dtype=torch.uint8, device=dev)
a_sf = torch.empty(256, padded_k, dtype=torch.int32, device=dev)
eo = torch.arange(0, E+1, dtype=torch.int32, device=dev)
torch.ops._C.mxfp4_experts_quant(a_q, a_sf, x.repeat(E,1), eo, eo, E)
a_sf8 = a_sf.view(torch.uint8)[:E*M]
s3 = s_gu.view(torch.uint8).view(E, 2*N, padded_k*4)
out = torch.empty(E*M, 2*N, dtype=torch.bfloat16, device=dev)
ps = torch.tensor([[M, 2*N, K]]*E, dtype=torch.int32, device=dev)
eo_lo = eo[:-1].contiguous()
us = bench(lambda: torch.ops._C.cutlass_mxfp4_group_mm(out, a_q, w_gu, a_sf8, s3, ps, eo_lo, eo_lo))
mb = E * 2*N * K/2 / 1e6
print(f"grouped MXFP4 (E=6,2N=4096,K=7168): {us:.1f} us -> {mb/us*1e3:.0f} GB/s")

# --- marlin ---
N2, K2 = 16384, 5120
w2 = torch.randint(0, 256, (N2, K2//2), dtype=torch.uint8, device=dev)
s2 = torch.full((N2, K2//32), 127, dtype=torch.uint8, device=dev)
x2 = torch.randn(1, K2, dtype=torch.bfloat16, device=dev)
from tinyinfer import gemv_fp4_marlin
gemv_fp4_marlin.gemv_fp4_fast(x2, w2, s2)
us2 = bench(lambda: gemv_fp4_marlin.gemv_fp4_fast(x2, w2, s2))
print(f"marlin mxfp4 (N=16384,K=5120):       {us2:.1f} us -> {N2*K2/2/1e6/us2*1e3:.0f} GB/s")
