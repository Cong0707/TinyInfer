import json
d = json.load(open(r"E:\Models\Qwen3.8-27B\config.json", encoding="utf-8"))
tc = d.get("text_config", d)
print("model_type:", d.get("model_type") or tc.get("model_type"))
print("architectures:", d.get("architectures"))
for k in ("hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
          "head_dim", "intermediate_size", "vocab_size", "max_position_embeddings",
          "tie_word_embeddings", "torch_dtype", "dtype"):
    if k in tc:
        print(f"  {k}: {tc[k]}")
lt = tc.get("layer_types")
if lt:
    print(f"  layer_types: {len(lt)} -> {sorted(set(lt))}")
q = tc.get("quantization_config") or d.get("quantization_config")
print("quantization:", q if q else "none (full precision)")
print()
g = json.load(open(r"E:\Models\Qwen3.8-27B\generation_config.json", encoding="utf-8"))
print("generation_config:", {k: v for k, v in g.items() if k != "transformers_version"})
