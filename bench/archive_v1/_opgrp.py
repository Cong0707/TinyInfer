
import torch, re
import tinyinfer._C_stable_libtorch, tinyinfer._moe_C_stable_libtorch  # noqa
ops = sorted(n for n in torch._C._dispatch_get_all_op_names()
             if n.startswith("_C::") or n.startswith("_moe_C::"))
print(f"算子总数: {len(ops)}")
# 按前缀归类，估算所需源文件
import collections
grp = collections.Counter()
for o in ops:
    n = o.split("::")[1]
    m = re.match(r"(fused_deepseek_v4|fused_kimi_k3|moe_|topk_|top_k_|marlin_|machete_|cutlass_|awq_|gptq_|nvfp4_|mxfp4_|scaled_|rms_|silu_|merge_|dsv3_|fp32_|sampling|apply_|attention|mla_|sm100_|selective_|gdn_|kda_|hadamard|per_token|persistent|dynamic_|minimax|grouped_|batched_|permute|sinkhorn|hc_|engram)", n)
    grp[m.group(1) if m else "other"] += 1
for k, v in sorted(grp.items(), key=lambda x: -x[1]):
    print(f"  {k:<22} {v}")
