import json
from pathlib import Path
GB = 2**30

# V4.1 actual
v = json.load(open(r"E:\Models\DeepSeek-V4.1-Flash-1017\config.json",
                    encoding="utf-8"))["text_config"]
print("=== DeepSeek-V4.1-Flash-1017 (本地真实 config) ===")
for k in ("num_hidden_layers","hidden_size","n_routed_experts","num_experts_per_tok",
          "n_shared_experts","moe_intermediate_size","num_attention_heads","head_dim",
          "q_lora_rank","o_lora_rank","kv_source_layer_ids","compress_ratios",
          "engram_layer_ids","hc_mult"):
    val = v.get(k)
    if isinstance(val, list) and len(val) > 8:
        val = f"{len(val)} entries"
    print(f"  {k}: {val}")
print()
print("=== 与 R1 (V3 架构) 的关键差异 ===")
print("  R1/V3: 61 层, 256 专家 top-8, MLA, 无压缩注意力")
print("  V4.1: 40 层, 384 专家 top-6, MLA + 压缩注意力 + engram + 超连接")
print()
ACTIVE = 13.5e9
print(f"V4.1 激活参数/token: {ACTIVE/1e9:.1f} B")
print("R1 激活参数/token  : 37 B")
print(f"-> V4.1 每 token 的 FLOPs 只有 R1 的 {ACTIVE/37e9*100:.0f}%")
print()
print("=== 所以 V4.1 应该比 R1 快得多，不是慢 ===")