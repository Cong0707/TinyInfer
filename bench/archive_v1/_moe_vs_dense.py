import json
GB = 2**30

print("=== DeepSeek-V4.1-Flash ===")
d = json.load(open(r"E:\Models\DeepSeek-V4.1-Flash-1017\config.json", encoding="utf-8"))
tc = d.get("text_config", d)
L = tc["num_hidden_layers"]; E = tc.get("n_routed_experts"); K = tc.get("num_experts_per_tok")
print(f"layers {L} | experts/layer {E} | top-k {K}")
print(f"ratio activated: {K}/{E} = {100*K/E:.2f}%")
total_b = sum(1 for _ in [])  # placeholder
shards = json.load(open(r"E:\Models\DeepSeek-V4.1-Flash-1017\model.safetensors.index.json",
                        encoding="utf-8"))["weight_map"]
import collections
sizes = collections.Counter()
for n, shard in shards.items():
    if ".experts." in n:
        sizes["expert"] += 1
    elif n.startswith("layers."):
        sizes["layer"] += 1
    else:
        sizes["other"] += 1
print("tensor groups:", dict(sizes))

print()
print("=== Qwen3.8-27B ===")
q = json.load(open(r"E:\Models\Qwen3.8-27B\config.json", encoding="utf-8"))
qtc = q.get("text_config", q)
print("moe fields present:", [k for k in qtc if "expert" in k.lower() or "moe" in k.lower()])
print(f"layers {qtc['num_hidden_layers']} | hidden {qtc['hidden_size']} | "
      f"intermediate {qtc['intermediate_size']}")
print("architecture: dense (every parameter is used for every token)")
