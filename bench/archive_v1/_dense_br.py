GB = 2**30; MB = 2**20
import json, struct
from pathlib import Path
from collections import defaultdict

MD = Path(r"E:\Models\DeepSeek-V4.1-Flash-1017")
idx = json.loads((MD/"model.safetensors.index.json").read_text(encoding="utf-8"))
wm = idx["weight_map"]; hd = {}
def hdr(s):
    if s not in hd:
        with (MD/s).open("rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hd[s] = json.loads(f.read(n))
    return hd[s]

# per-layer dense breakdown, excluding experts
by = defaultdict(int)
for name, shard in wm.items():
    if not name.startswith("layers."): continue
    parts = name.split(".")
    lid = parts[1]
    rest = ".".join(parts[2:])
    if rest.startswith("ffn.experts"): continue
    if rest.startswith("engram.embed"): continue
    e = hdr(shard).get(name)
    if e is None: continue
    b = e["data_offsets"][1]-e["data_offsets"][0]
    by[rest] += b

print("per-layer dense tensors (summed over 40 layers, then /40):")
tot = 0
for k, b in sorted(by.items(), key=lambda kv:-kv[1]):
    print(f"  {k:36s} {b/40/MB:7.2f} MB/layer")
    tot += b
print(f"  {"TOTAL dense":36s} {tot/40/MB:7.2f} MB/layer")
print(f"  all 40 layers: {tot/GB:.2f} GB")