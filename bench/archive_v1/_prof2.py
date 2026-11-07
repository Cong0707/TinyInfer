
import sys
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.loading import ShardIndex
from tinyinfer.resources import _layer_of

GB = 1 << 30
idx = ShardIndex(r"E:/models/DeepSeek-V4.1-Flash-1017")
wm = idx.weight_map
header = idx._header
per_layer, n_skip, n_miss = {}, 0, 0
for name in wm:
    lid = _layer_of(name)
    if lid is None:
        n_skip += 1
        continue
    if "experts" in name.split("."):
        n_skip += 1
        continue
    try:
        parsed = header(wm[name])
        tensors = parsed[0] if isinstance(parsed, tuple) else parsed
        ent = tensors.get(name)
        if ent is None:
            n_miss += 1
            continue
        n = ent["data_offsets"][1] - ent["data_offsets"][0]
    except Exception:
        n_miss += 1
        continue
    per_layer[lid] = per_layer.get(lid, 0) + int(n)

print("tensors:", len(wm), "skip:", n_skip, "miss:", n_miss, "layers:", len(per_layer))
top = sorted(per_layer.items(), key=lambda kv: -kv[1])[:6]
for lid, b in top:
    print(f"  layer {lid:3d}: {b/GB:8.3f} GB")
print("avg:", sum(per_layer.values())/len(per_layer)/GB, "GB")
