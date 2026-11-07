print("=== tinyinfer 的 kernel 清单 ===")
from pathlib import Path
ours = sorted(p.stem for p in Path(r"E:\Develop\Deepseek\tinyinfer\src\tinyinfer\native").glob("*.cu"))
for o in ours:
    print(f"  {o}")
print()
print("=== vLLM 对应项 ===")
map_ = {
  "gemv_bf16": "无直接对应（vLLM 用 cutlass scaled_mm）",
  "gemv_fp4_e8m0": "quantization/marlin (W4A16) / quantization/fp4/*",
  "head_gemv": "无（vLLM 用普通 GEMM）",
  "moe_gemv6": "moe/moe_wna16.cu",
  "moe_w13": "moe/marlin_moe_wna16/ops.cu (fused w1+w3)",
  "moe_act": "activation_kernels.cu / moe/swiglu",
  "rms_norm": "layernorm_kernels.cu",
  "qwen_norm": "(Qwen 专用 RMSNorm)",
  "qwen_fused": "(Qwen 融合)",
  "gdn": "gdn/fused_gdn_decode_kernel.cu",
  "hc": "hyper-connections（V4 专用，vLLM 无）",
  "sinkhorn": "无",
}
for k, v in map_.items():
    print(f"  {k:16s} -> {v}")
print()
print("=== vLLM 有而我们完全没有的 ===")
missing = [
  "fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu (67KB)",
  "cache_kernels_fused.cu (KV 写入融合)",
  "moe/moe_align_sum_kernels.cu (49KB, MoE 对齐)",
  "moe/moe_permute_unpermute_op.cu (专家重排)",
  "moe/grouped_topk_kernels.cu (59KB, 分组 topk)",
  "quantization/fp4/mxfp4_blockwise_moe_kernel.cu",
  "quantization/cutlass_w4a8/* (W4A8 分组 GEMM)",
  "sampler.cu (41KB, 采样器)",
  "activation_kernels.cu (64KB, 所有激活)",
  "custom_all_reduce.cu (多卡通信)",
]
for m in missing:
    print(f"  {m}")