import json, os
from pathlib import Path
from collections import defaultdict

MD = Path(r"E:\Models\DeepSeek-V4.1-Flash-1017")
d = json.load(open(MD / "config.json", encoding="utf-8"))
tc = d.get("text_config", d)
qc = d.get("quantization_config", {})

print("=== DeepSeek-V4.1-Flash-1017 ===")
for k in ("num_hidden_layers", "hidden_size", "n_routed_experts", "num_experts_per_tok",
          "n_shared_experts", "moe_intermediate_size", "intermediate_size",
          "num_attention_heads", "kv_lora_rank", "q_lora_rank", "qk_rope_head_dim",
          "qk_nope_head_dim", "v_head_dim", "vocab_size", "max_position_embeddings"):
    if k in tc:
        print(f"  {k}: {tc[k]}")
print("  quantization_config:", {k: v for k, v in qc.items() if k != "modules_to_not_convert"})
print()

# exact tensor bytes from the index + shard headers
idx = json.loads((MD / "model.safetensors.index.json").read_text(encoding="utf-8"))
wm = idx["weight_map"]
hdrs = {}
def hdr(shard):
    if shard not in hdrs:
        import struct
        with (MD / shard).open("rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdrs[shard] = json.loads(f.read(n))
    return hdrs[shard]

groups = defaultdict(int)
for name, shard in wm.items():
    h = hdr(shard)
    e = h.get(name)
    if e is None:
        continue
    nbytes = e["data_offsets"][1] - e["data_offsets"][0]
    if ".experts." in name:
        g = "routed_experts"
    elif "shared_experts" in name:
        g = "shared_experts"
    elif name.startswith("layers."):
        g = "layer_other"
    else:
        g = "embed_head_norm"
    groups[g] += nbytes

total = sum(groups.values())
print("total checkpoint bytes by group:")
for g, b in sorted(groups.items(), key=lambda kv: -kv[1]):
    print(f"  {g:18s} {b/2**30:7.2f} GB")
print(f"  {'TOTAL':18s} {total/2**30:7.2f} GB")
print()

# one expert blob
ex = [n for n in wm if ".experts." in n]
per_layer_experts = defaultdict(int)
for n in ex:
    parts = n.split(".")
    for i, p in enumerate(parts):
        if p == "layers":
            per_layer_experts[int(parts[i+1])] += 0
            break
lids = set()
for n in ex:
    parts = n.split(".")
    for i, p in enumerate(parts):
        if p == "layers":
            lids.add(int(parts[i+1])); break
E = tc["n_routed_experts"]; L = tc["num_hidden_layers"]
print(f"expert tensors: {len(ex)} | layers with experts: {len(lids)}/{L} | experts/layer: {E}")
print(f"  -> bytes per expert = {groups['routed_experts']/(len(lids)*E)/2**20:.2f} MB")
print(f"  -> bytes per layer (non-expert) = {groups['layer_other']/len(lids)/2**20:.2f} MB")
