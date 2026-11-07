"""Map every vLLM kernel to what tinyinfer does today, so the baseline is explicit."""
from pathlib import Path
root = Path(r"E:\Develop\Deepseek\repositories\vllm-csrc\csrc")

groups = {
 "量化 GEMM (fp4/fp8/int4)": [],
 "MoE (路由/对齐/GEMM)": [],
 "注意力 (MLA/decode)": [],
 "KV cache 写入": [],
 "norm / rope / 融合": [],
 "激活 / 采样": [],
 "DeepSeek 专用": [],
 "通信 / 其他": [],
}
for f in root.rglob("*.cu"):
    rel = str(f.relative_to(root))
    n = rel.lower()
    if "rocm" in n:
        continue
    if any(k in n for k in ("marlin", "fp4", "fp8", "int4", "quant", "q_gemm", "w8a8", "cutlass")):
        groups["量化 GEMM (fp4/fp8/int4)"].append(rel)
    elif "moe" in n or "topk" in n or "expert" in n:
        groups["MoE (路由/对齐/GEMM)"].append(rel)
    elif "attention" in n or "mla" in n or "flash" in n:
        groups["注意力 (MLA/decode)"].append(rel)
    elif "cache" in n:
        groups["KV cache 写入"].append(rel)
    elif any(k in n for k in ("norm", "rope", "layernorm", "rms")):
        groups["norm / rope / 融合"].append(rel)
    elif any(k in n for k in ("activation", "silu", "gelu", "sampler", "top_k", "top_p")):
        groups["激活 / 采样"].append(rel)
    elif "deepseek" in n or "dsv" in n:
        groups["DeepSeek 专用"].append(rel)
    else:
        groups["通信 / 其他"].append(rel)

for g, items in groups.items():
    if not items:
        continue
    print(f"\n## {g}  ({len(items)})")
    for i in sorted(items)[:12]:
        print(f"   {i}")
    if len(items) > 12:
        print(f"   ... +{len(items)-12}")
