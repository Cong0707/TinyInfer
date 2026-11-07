import json, struct
from pathlib import Path
from collections import defaultdict
GB = 2**30; MB = 2**20

MD = Path(r"E:\Models\DeepSeek-V4.1-Flash-1017")
d = json.load(open(MD / "config.json", encoding="utf-8")); tc = d.get("text_config", d)
idx = json.loads((MD / "model.safetensors.index.json").read_text(encoding="utf-8"))
wm = idx["weight_map"]; hdrs = {}
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
    if "engram.embed" in name:
        cat = "engram_tables"      # giant lookup matrices, read by row only
    elif "ffn.experts." in name:
        cat = "routed_experts"
    elif "shared_experts" in name:
        cat = "shared_experts"
    elif "mtp." in name or "aligner." in name:
        cat = "mtp_vision"
    elif name.startswith("layers."):
        cat = "layer_dense"
    else:
        cat = "embed_head"
    g[cat] += n

W = sum(g.values())
print("=== exact weight budget ===")
for k, b in sorted(g.items(), key=lambda kv: -kv[1]):
    print(f"  {k:16s} {b/GB:8.2f} GB")
print(f"  {'TOTAL':16s} {W/GB:8.2f} GB")
print()
print(f"engram tables: {g['engram_tables']/GB:.1f} GB, but a token reads only a few rows")
print(f"  -> resident weights a GPU actually needs: {(W-g['engram_tables'])/GB:.1f} GB")
print()

L = tc["num_hidden_layers"]; E = tc["n_routed_experts"]; TOPK = tc["num_experts_per_tok"]
per_expert = g["routed_experts"]/(L*E)
dense_l = g["layer_dense"]/L
print(f"per layer dense {dense_l/MB:.0f} MB | expert {per_expert/MB:.2f} MB x{E}")
print()
dense_all = L*dense_l
experts_tok = L*TOPK*per_expert
tables = g["embed_head"]
engram_row = 8*MB   # a handful of rows per token, negligible
per_token = dense_all + experts_tok + tables + engram_row
print("=== per-token weight reads (all resident in HBM) ===")
print(f"  40 dense layers   : {dense_all/GB:7.2f} GB  ({100*dense_all/per_token:.0f}%)")
print(f"  240 experts       : {experts_tok/GB:7.2f} GB  ({100*experts_tok/per_token:.0f}%)")
print(f"  embed + head      : {tables/GB:7.2f} GB")
print(f"  TOTAL             : {per_token/GB:7.2f} GB/token")
print()
PER_CARD_GB = 141.0; PER_CARD_TBS = 4.8
print("=== decode floor on H200 (bandwidth bound, all weights resident) ===")
for n in (4, 8, 16):
    cap = n*PER_CARD_GB
    room = cap - (W-g["engram_tables"])/GB
    agg = n*PER_CARD_TBS
    ms = per_token/GB/agg*1000
    print(f"  {n:2d} x H200: {cap:5.0f} GB weights-room {room:+6.1f} GB | "
          f"{agg:4.1f} TB/s | {ms:6.1f} ms/token | {1000/ms:5.1f} tok/s")
