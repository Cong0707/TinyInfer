
import torch
from tinyinfer import vllm_ops as vo
vo._ops()
moe = vo._moe_ops()
names = sorted(x for x in dir(moe) if not x.startswith("_"))
print("moe ops:", len(names))
for n in ("topk_softplus_sqrt", "grouped_topk", "topk_softmax", "moe_align_block_size",
          "moe_wna16_marlin_gemm", "moe_sum", "moe_permute", "moe_unpermute",
          "cutlass_mxfp4_group_mm", "cutlass_fp4_group_mm", "fused_marlin_moe"):
    print(" ", n, "=", n in names)
print("all:", names)
