
import torch
import tinyinfer._C_stable_libtorch  # noqa
import tinyinfer._moe_C_stable_libtorch  # noqa

for n in ("moe_wna16_marlin_gemm", "moe_wna16_gemm", "moe_align_block_size",
          "moe_permute", "moe_unpermute", "moe_sum"):
    op = getattr(torch.ops._moe_C, n)
    print("###", n)
    for s in op._schemas:
        print("   ", str(s)[:300])
    print()

print("=== ScalarType ids ===")
try:
    from vllm.scalar_type import scalar_types
    for k in ("float4_e2m1f", "float4_e2m1fn_x2", "uint4b8", "uint8b128"):
        print("  ", k, getattr(scalar_types, k, None))
except Exception as exc:
    print("  vllm.scalar_type unavailable:", exc)
