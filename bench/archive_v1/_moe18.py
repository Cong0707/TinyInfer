
import torch
import tinyinfer._C_stable_libtorch  # noqa
import tinyinfer._moe_C_stable_libtorch  # noqa

names = torch._C._dispatch_get_all_op_names()
m = sorted(n.split("::")[1] for n in names if n.startswith("_moe_C::"))
print("_moe_C ops (%d):" % len(m))
for n in m:
    print("  ", n)

print()
print("=== device capability:", torch.cuda.get_device_capability())
for name in ("cutlass_group_gemm_supported", "cutlass_scaled_mm_supports_fp4",
             "cutlass_scaled_mm_supports_fp8", "cutlass_scaled_mm_supports_block_fp8"):
    op = getattr(torch.ops._C, name, None)
    try:
        print(f"  {name}() =", op() if op is not None else "MISSING")
    except Exception as exc:
        print(f"  {name}() raised {type(exc).__name__}: {str(exc)[:120]}")
