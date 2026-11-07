import json
from pathlib import Path

d = Path(r"E:\Develop\Deepseek\tinyinfer\cache\weights")
v = json.loads((d / "verified.json").read_text(encoding="utf-8"))
shards = set(v.keys()) if isinstance(v, dict) else set(v)
idx = json.loads((d / "model.safetensors.index.json").read_text(encoding="utf-8"))
wm = idx["weight_map"]
# layer -> its attn shard
layer_shard = {}
for name, shard in wm.items():
    if name.startswith("layers.") and ".attn." in name:
        layer = int(name.split(".")[1])
        layer_shard.setdefault(layer, shard)
complete = sorted(l for l, s in layer_shard.items() if s in shards)
print("verified shards:", sorted(shards)[:8], "...")
print("complete layers:", complete[:5], "...", complete[-5:], "count:", len(complete))
