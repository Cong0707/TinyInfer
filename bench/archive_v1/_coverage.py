
import torch, re
import tinyinfer._C_stable_libtorch  # noqa
import tinyinfer._moe_C_stable_libtorch  # noqa
ops = sorted(n.split("::")[1] for n in torch._C._dispatch_get_all_op_names()
             if n.startswith("_C::") or n.startswith("_moe_C::"))
# vLLM 支持但这里没有的关键能力
want = {
 "PagedAttention/FlashAttn": r"paged_attention|flash_attn|reshape_and_cache|_vllm_fa|flash_attn_varlen",
 "KV cache 管理": r"reshape_and_cache|cache_ops|copy_blocks|swap_blocks",
 "采样": r"sampling|sample_|penalties|top_k_|top_p_|temperature",
 "Mamba/线性注意力": r"mamba|ssm|causal_conv|selective_scan|gdn|kda",
 "量化GEMM": r"marlin|machete|cutlass_scaled_mm|fp4|fp8|w4a8|gptq|awq",
 "MoE": r"moe|topk|top_k|expert|align_block",
 "MLA": r"mla",
 "激活/归一化": r"norm|silu|act|swiglu|clamp|gelu",
 "RoPE": r"rope|rotary|qk_norm",
}
print("=== 实际存在的算子（按类别）===")
for k, pat in want.items():
    hit = [o for o in ops if re.search(pat, o, re.I)]
    print(f"{k:<24} {len(hit):>3}  {', '.join(hit[:4])}{' ...' if len(hit)>4 else ''}")
