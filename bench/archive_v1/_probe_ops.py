
import torch
import vllm._C_stable_libtorch as C  # noqa
import vllm._moe_C_stable_libtorch as M  # noqa

for name in ("gptq_gemm", "awq_gemm", "machete_gemm", "cutlass_scaled_mm", "cutlass_scaled_mm_azp", "scaled_fp8_quant", "reshape_and_cache", "paged_attention_v1", "flash_attn_varlen_func", "merge_attn_states", "fused_deepseek_v4_qnorm_rope_kv_insert", "moe_align_block_size", "get_moe_activation", "topk_softmax", "grouped_topk", "mxfp4_experts_quant", "scaled_fp4_quant"):
    try:
        getattr(torch.ops._C, name)
        print("C  OK ", name)
    except Exception:
        pass
for name in ("topk_softmax", "moe_align_block_size", "fused_topk", "grouped_topk", "moe_sum", "get_moe_activation", "moe_permute", "moe_unpermute", "topk_softmax_with_capacity", "moe_lora_align_block_size"):
    try:
        getattr(torch.ops._moe_C, name)
        print("MOE OK", name)
    except Exception:
        pass
