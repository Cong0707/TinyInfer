
import sys
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.loading import ShardIndex
from tinyinfer.resources import _layer_of

GB = 1 << 30
idx = ShardIndex(r"E:/models/DeepSeek-V4.1-Flash-1017")
wm, header = idx.weight_map, idx._header
for target in (1, 14, 8):
    print(f"--- layer {target} ---")
    rows = []
    for name in wm:
        if _layer_of(name) != target or "experts" in name.split("."):
            continue
        parsed = header(wm[name])
        tensors = parsed[0] if isinstance(parsed, tuple) else parsed
        ent = tensors.get(name)
        if ent is None:
            continue
        n = ent["data_offsets"][1] - ent["data_offsets"][0]
        rows.append((n, name.split(".")[-2:], wm[name]))
    rows.sort(reverse=True)
    for n, tail, shard in rows[:5]:
        print(f"   {n/GB:9.3f} GB  '{'.'.join(tail)}'  <- {shard}")
    print(f"   count={len(rows)} total={sum(r[0] for r in rows)/GB:.3f} GB")
