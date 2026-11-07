import json
from pathlib import Path
idx = json.loads(Path(r"E:\Models\Qwen3.8-27B\model.safetensors.index.json").read_text())
wm = idx["weight_map"]
# how many distinct shards does one layer touch?
import re, collections
for layer in (0, 1, 2, 3, 63):
    keys = [k for k in wm if f"layers.{layer}." in k]
    shards = sorted({wm[k] for k in keys})
    print(f"layer {layer}: {len(keys)} tensors across {len(shards)} shards")
    if layer == 0:
        for k in keys[:6]:
            print("   ", k, "->", wm[k])
print()
print("total tensors:", len(wm))
print("total shards:", len(set(wm.values())))
