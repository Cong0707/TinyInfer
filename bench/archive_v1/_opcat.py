
import torch, collections, re
import tinyinfer._C_stable_libtorch  # noqa
import tinyinfer._moe_C_stable_libtorch  # noqa
ops = sorted(n for n in torch._C._dispatch_get_all_op_names()
             if n.startswith("_C::") or n.startswith("_moe_C::"))
print("total:", len(ops))
cats = collections.Counter()
for o in ops:
    n = o.split("::")[1]
    if re.search(r"marlin|machete|gptq|awq|w4a8|mxfp4|nvfp4|fp4|fp8|quant|scaled", n): c="量化/低精度GEMM"
    elif re.search(r"moe|topk|top_k|expert|permute|align", n): c="MoE/路由"
    elif re.search(r"mla|attn|attention|cache|kv|rope|qk", n): c="注意力/KV/位置"
    elif re.search(r"norm|silu|act|mul|add|gelu|swiglu|clamp", n): c="逐元素/激活"
    elif re.search(r"mamba|ssm|gd|kda|selective|scan|delta", n): c="Mamba/线性注意力"
    elif re.search(r"sampl|logit|penal|top_p|temperature", n): c="采样"
    elif re.search(r"hadamard|fft", n): c="变换"
    else: c="其他"
    cats[c]+=1
for c,n in cats.most_common(): print(f"  {c:<18} {n}")
