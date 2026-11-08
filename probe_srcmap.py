import json, sys
cfg = json.load(open("/models/deepseek/config.json", encoding="utf-8"))["text_config"]
srcs = cfg["kv_source_layer_ids"]
ratios = cfg["compress_ratios"][:40]
print("kv_source_layer_ids", srcs, flush=True)
print("src ratios", [ratios[i] for i in srcs], flush=True)
# 每层应该跟哪个 source
groups = {}
for l in range(40):
    own = [s for s in srcs if s <= l]
    g = own[-1] if own else None
    groups[l] = (g, ratios[g] if g is not None else ratios[l])
print("layer -> (source, ratio):", flush=True)
for l in range(0, 40, 6):
    print("  L%d -> %s" % (l, groups[l]), flush=True)
print("distinct ratios used:", sorted(set(r for _, r in groups.values())), flush=True)