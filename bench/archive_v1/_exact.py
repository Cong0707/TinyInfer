import json, struct
from pathlib import Path
from collections import defaultdict
GB = 2**30; MB = 2**20

MD = Path(r"E:\Models\DeepSeek-V4.1-Flash-1017")
d = json.load(open(MD / "config.json", encoding="utf-8"))
tc = d.get("text_config", d)
idx = json.loads((MD / "model.safetensors.index.json").read_text(encoding="utf-8"))
wm = idx["weight_map"]
hdrs = {}
def hdr(shard):
    if shard not in hdrs:
        with (MD / shard).open("rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdrs[shard] = json.loads(f.read(n))
    return hdrs[shard]

g = defaultdict(int)
for name, shard in wm.items():
    h = hdr(shard); e = h.get(name)
    if e is None: continue
    n = e["data_offsets"][1] - e["data_offsets"][0]
    if "ffn.experts." in name:
        cat = "routed_experts"
    elif "shared_experts" in name:
        cat = "shared_experts"
    elif "mtp." in name or "aligner." in name:
        cat = "mtp_vision"
    elif name.startswith("layers."):
        cat = "layer_dense"      # attention + gate + shared + hc + norms
    else:
        cat = "embed_head"
    g[cat] += n

W = sum(g.values())
print("=== DeepSeek-V4.1-Flash-1017, exact weights ===")
for k, b in sorted(g.items(), key=lambda kv: -kv[1]):
    print(f"  {k:16s} {b/GB:7.2f} GB")
print(f"  {'TOTAL':16s} {W/GB:7.2f} GB")
print()

L = tc["num_hidden_layers"]; E = tc["n_routed_experts"]; TOPK = tc["num_experts_per_tok"]
per_expert = g["routed_experts"] / (L * E)
dense_per_layer = g["layer_dense"] / L
print(f"per layer: dense {dense_per_layer/MB:.0f} MB | {E} experts x "
      f"{per_expert/MB:.2f} MB = {E*per_expert/MB:.0f} MB experts")
print(f"per token reads: dense {L*dense_per_layer/MB:.0f} MB + "
      f"{L} x {TOPK} x {per_expert/MB:.2f} MB = {L*TOPK*per_expert/MB:.0f} MB")
print()

# ---- per-token working set ----------------------------------------------
dense_all = L * dense_per_layer
experts_hot = L * TOPK * per_expert
tables = g["embed_head"]
per_token = dense_all + experts_hot + tables
print("=== bytes a token must read ===")
print(f"  all 40 dense layers : {dense_all/GB:6.1f} GB")
print(f"  240 routed experts  : {experts_hot/GB:6.1f} GB")
print(f"  embed + head        : {tables/GB:6.1f} GB")
print(f"  TOTAL               : {per_token/GB:6.1f} GB/token")
print()
print(f"  (of which the experts are only {100*experts_hot/per_token:.0f}% -- this")
print(f"   architecture is NOT expert-dominated for a single token)")
