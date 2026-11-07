import json, struct
from pathlib import Path
from collections import defaultdict
MB = 2**20
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

# every distinct tensor-name suffix across all layers, summed
by_suffix = defaultdict(lambda: [0, 0])
for name, shard in wm.items():
    if not name.startswith("layers."): continue
    parts = name.split(".")
    # strip "layers.<i>."
    suffix = ".".join(parts[2:])
    h = hdr(shard); e = h.get(name)
    if e is None: continue
    n = e["data_offsets"][1] - e["data_offsets"][0]
    by_suffix[suffix][0] += n
    by_suffix[suffix][1] += 1

print("largest tensor groups inside layers (sum over all 40 layers):")
rows = sorted(by_suffix.items(), key=lambda kv: -kv[1][0])[:12]
for suf, (b, c) in rows:
    print(f"  {suf:34s} {b/MB:9.1f} MB  x{c}")
print()
# how many of these are expert tensors?
ex = sum(b for suf, (b, c) in by_suffix.items() if suf.startswith("ffn.experts"))
tot = sum(b for b, c in by_suffix.values())
print(f"experts inside layers : {ex/MB:9.1f} MB  ({100*ex/tot:.1f}% of layer bytes)")
print(f"non-expert inside     : {(tot-ex)/MB:9.1f} MB")
print(f"per layer non-expert  : {(tot-ex)/40/MB:9.1f} MB")
