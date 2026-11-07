
import torch
print("=== only _moe_C ===")
try:
    import tinyinfer._moe_C_stable_libtorch  # noqa
    ops = sorted(x for x in dir(torch.ops._moe_C) if not x.startswith("_"))
    print("moe ops:", len(ops), ops[:15])
except Exception as exc:
    print("import failed:", type(exc).__name__, str(exc)[:200])
    ops = sorted(x for x in dir(torch.ops._moe_C) if not x.startswith("_"))
    print("moe ops after failure:", len(ops), ops[:15])
print()
print("=== does _C also carry moe ops? ===")
import tinyinfer._C_stable_libtorch  # noqa
c_ops = [x for x in dir(torch.ops._C) if not x.startswith("_")]
print("_C op count:", len(c_ops))
for n in ("topk_softmax", "topk_softplus_sqrt", "grouped_topk", "moe_align_block_size",
          "moe_sum", "moe_permute", "moe_unpermute", "marlin_gemm", "silu_and_mul"):
    print("  _C.", n, "=", n in c_ops)
