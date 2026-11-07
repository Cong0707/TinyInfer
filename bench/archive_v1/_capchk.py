
import torch
import tinyinfer._C_stable_libtorch  # noqa
import tinyinfer._moe_C_stable_libtorch  # noqa

print("=== _moe_C attribute access ===")
for n in ["moe_align_block_size", "moe_permute", "moe_unpermute", "moe_sum",
          "moe_wna16_marlin_gemm", "topk_softmax", "topk_softplus_sqrt", "grouped_topk"]:
    try:
        getattr(torch.ops._moe_C, n)
        print("  OK  ", n)
    except Exception as exc:
        print("  FAIL", n, type(exc).__name__, str(exc)[:80])

print()
print("=== capability support ===")
for cap in ("12.0", (12, 0), 120):
    for fn in ("cutlass_scaled_mm_supports_fp4", "cutlass_group_gemm_supported"):
        op = getattr(torch.ops._C, fn, None)
        try:
            print(f"  {fn}({cap!r}) =", op(cap) if op else "MISSING")
        except Exception as exc:
            print(f"  {fn}({cap!r}) raised: {type(exc).__name__} {str(exc)[:90]}")
