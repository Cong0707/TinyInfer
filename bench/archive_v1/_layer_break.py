import json, struct
from pathlib import Path
from collections import defaultdict

MD = Path(r"E:\Models\DeepSeek-V4.1-Flash-1017")
idx = json.loads((MD / "model.safetensors.index.json").read_text(encoding="utf-8"))
wm = idx["weight_map"]
hdrs = {}
def hdr(shard):
    if shard not in hdrs:
        with (MD / shard).open("rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdrs[shard] = json.loads(f.read(n))
    return hdrs[shard]

# break down one layer's non-expert tensors
target = "layers.0."
buckets = defaultdict(int)
counts = defaultdict(int)
for name, shard in wm.items():
    if not name.startswith(target):
        continue
    h = hdr(shard); e = h.get(name)
    if e is None: continue
    n = e["data_offsets"][1] - e["data_offsets"][0]
    rest = name[len(target):]
    key = ".".join(rest.split(".")[:2])
    buckets[key] += n
    counts[key] += 1

print("layer 0 non-expert tensors by component:")
tot = 0
for k, b in sorted(buckets.items(), key=lambda kv: -kv[1]):
    print(f"  {k:28s} {b/2**20:8.1f} MB  ({counts[k]} tensors)")
    tot += b
print(f"  {'TOTAL':28s} {tot/2**20:8.1f} MB")
print()

# and what the top-level non-layer tensors are
print("top-level tensors:")
tl = defaultdict(int)
for name, shard in wm.items():
    if name.startswith("layers."): continue
    h = hdr(shard); e = h.get(name)
    if e is None: continue
    tl[name] = e["data_offsets"][1] - e["data_offsets"][0]
for k, b in sorted(tl.items(), key=lambda kv: -kv[1])[:8]:
    print(f"  {k:40s} {b/2**20:8.1f} MB")
