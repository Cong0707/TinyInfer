GB = 2**30; MB = 2**20
import json

tc = json.load(open(r"E:\Models\DeepSeek-V4.1-Flash-1017\config.json",
                    encoding="utf-8"))["text_config"]
print("=== 重新核对 MLA 的 KV 结构 ===")
for k in ("head_dim", "qk_rope_head_dim", "num_attention_heads",
          "kv_source_layer_ids", "compress_ratios", "index_head_dim"):
    print(f"  {k}: {tc[k]}")
print()

# MLA stores ONE compressed latent per token, not one per head.
# latent size = head_dim (512) which already includes the rope part.
L = tc["num_hidden_layers"]
hd = tc["head_dim"]
kvs = tc["kv_source_layer_ids"]
comp = tc["compress_ratios"]

print("Which layers have their own KV window?")
for i in range(L):
    mark = " OWN" if i in kvs else ""
    if i < 24 or mark:
        print(f"  layer {i:2d}: compress_ratio {comp[i]}{mark}")
print(f"  ... ({L} layers total)")
print()

print("=== KV bytes per token, per sequence ===")
print("Only the OWNING layers store a window; followers read the owner via")
print("kv_source_layer_ids. So the count is len(kv_source_layer_ids), not L.")
for label, b in (("fp16", 2), ("fp8", 1)):
    tot = 0
    for lid in kvs:
        r = max(1, comp[lid])
        tot += hd * b / r
    print(f"  {label}: {tot:7.0f} bytes/token")
    for ctx in (131072, 1048576):
        print(f"     ctx {ctx:>9,}: {tot*ctx/GB:7.3f} GB")