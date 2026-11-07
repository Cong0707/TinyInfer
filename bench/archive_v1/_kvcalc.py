import sys, json
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
import torch
# KV per position for Qwen3.5-0.8B: 6 full-attention layers, 2 kv heads, head_dim 256, bf16
cfg = json.load(open(r"E:\Models\Qwen3.5-0.8B\config.json", encoding="utf-8"))["text_config"]
n_full = sum(1 for t in cfg["layer_types"] if t == "full_attention")
kvh = cfg["num_key_value_heads"]; hd = cfg["head_dim"]
per_pos = n_full * 2 * kvh * hd * 2   # k and v, bf16
print(f"full-attention layers: {n_full}, kv heads: {kvh}, head_dim: {hd}")
print(f"KV bytes per position: {per_pos}  ({per_pos/1024:.1f} KB)")
free = torch.cuda.mem_get_info()
print(f"free now: {free[0]/2**30:.2f} GB, total {free[1]/2**30:.2f} GB")
for tok in (8192, 32768, 65536, 131072, 262144):
    print(f"  {tok:7d} tok -> {tok*per_pos/2**30:7.2f} GB KV")
