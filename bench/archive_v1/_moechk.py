
import torch
from tinyinfer import vllm_ops as vo

print("available:", vo.available())
names_moe = ["moe_wna16_marlin_gemm", "moe_align_block_size", "marlin_gemm",
             "moe_sum", "moe_permute", "moe_unpermute", "topk_softplus_sqrt",
             "grouped_topk", "topk_softmax"]
for n in names_moe:
    print(f"  _moe_C.{n}:", hasattr(torch.ops._moe_C, n))
print()
for n in ["marlin_gemm", "moe_align_block_size", "moe_sum", "moe_permute",
          "moe_unpermute", "silu_and_mul", "rms_norm"]:
    print(f"  _C.{n}:", hasattr(torch.ops._C, n))
if hasattr(torch.ops._moe_C, "moe_wna16_marlin_gemm"):
    print("\nschema:", torch.ops._moe_C.moe_wna16_marlin_gemm._schemas)
