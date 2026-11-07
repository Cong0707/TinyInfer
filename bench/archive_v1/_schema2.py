
import torch
import vllm._C_stable_libtorch  # noqa
import vllm._moe_C_stable_libtorch  # noqa

names_c = ["gptq_gemm", "awq_gemm", "cutlass_scaled_mm", "merge_attn_states", "mxfp4_experts_quant"]
names_m = ["topk_softmax", "moe_align_block_size", "grouped_topk", "moe_sum", "moe_permute", "moe_unpermute"]
for ns, names in ((torch.ops._C, names_c), (torch.ops._moe_C, names_m)):
    for n in names:
        op = getattr(ns, n)
        for ov in op.overloads():
            print(f"{n}.{ov}: {getattr(op, ov)._schema}")
            break
