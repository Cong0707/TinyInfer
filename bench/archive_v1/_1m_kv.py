GB = 2**30; MB = 2**20
import json

tc = json.load(open(r"E:\Models\DeepSeek-V4.1-Flash-1017\config.json",
                    encoding="utf-8"))["text_config"]
L = tc["num_hidden_layers"]
hd = tc["head_dim"]       # 512 for MLA latent
kvs = tc["kv_source_layer_ids"]
comp = tc["compress_ratios"]
print(f"layers {L} | MLA head_dim {hd} | KV-owning layers {kvs}")
print(f"compress_ratios for those layers: {[comp[i] for i in kvs]}")
print()

# MLA: one latent per token per owning layer, not per head
def kv_per_seq(ctx, bytes_per_val):
    tot = 0
    for lid in kvs:
        r = max(1, comp[lid]) if lid < len(comp) else 1
        tot += ctx * hd * bytes_per_val / r
    return tot

CTX = 1048576
print(f"=== KV per sequence at {CTX:,} tokens ===")
for name, b in (("fp16", 2), ("fp8", 1), ("4bit", 0.5)):
    v = kv_per_seq(CTX, b)
    print(f"  {name:5s}: {v/GB:7.3f} GB  ({v/MB:8.0f} MB)")
print()

W_RESIDENT = 286.4
for n, cap in ((4, 564.0), (8, 1128.0)):
    room = cap - W_RESIDENT
    print(f"=== {n} x H200, {cap:.0f} GB total, KV room {room:.0f} GB ===")
    for name, b in (("fp16", 2), ("fp8", 1)):
        per = kv_per_seq(CTX, b) / GB
        print(f"  {name:5s}: {per:6.3f} GB/seq -> {int(room/per):5d} concurrent @1M ctx")
    print()