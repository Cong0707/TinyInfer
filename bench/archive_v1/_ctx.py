import json
d = json.load(open(r"E:\Models\Qwen3.5-0.8B\config.json", encoding="utf-8"))
tc = d.get("text_config", d)
for k in ("max_position_embeddings", "rope_parameters", "sliding_window", "layer_types"):
    v = tc.get(k)
    if k == "layer_types":
        v = f"{len(v)} layers: {sorted(set(v))}"
    print(f"{k}: {v}")
