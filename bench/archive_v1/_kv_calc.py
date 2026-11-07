"""KV cost + the actual bill of materials."""
GB = 2**30; MB = 2**20
import json
from pathlib import Path

d = json.load(open(r"E:\Models\DeepSeek-V4.1-Flash-1017\config.json", encoding="utf-8"))
tc = d.get("text_config", d)
L = tc["num_hidden_layers"]; kvl = tc["kv_lora_rank"]; rope = tc["qk_rope_head_dim"]
print(f"MLA: kv_lora_rank={kvl}, qk_rope_head_dim={rope} -> per token per layer "
      f"{kvl+rope} values")
print(f"  40 layers x {kvl+rope} x 1 byte (fp8) = {L*(kvl+rope)/1024:.2f} KB/token")
print(f"  at 128K context: {L*(kvl+rope)*131072/GB:.2f} GB per sequence")
print(f"  at 1M context  : {L*(kvl+rope)*1048576/GB:.2f} GB per sequence")
print()

W_RESIDENT = 286.4      # GB, excludes the engram lookup tables
PER_TOKEN = 12.93
HBM = 141.0; TBS = 4.8
print("=== configurations ===")
for n in (4, 8):
    cap = n * HBM
    room = cap - W_RESIDENT
    agg = n * TBS
    ms = PER_TOKEN / agg * 1000
    print(f"\n{n} x H200 ({cap:.0f} GB, {agg:.1f} TB/s aggregate)")
    print(f"  weights          : {W_RESIDENT:6.1f} GB")
    print(f"  left for KV      : {room:6.1f} GB")
    per_seq = L*(kvl+rope)*131072/GB
    print(f"  -> {int(room/per_seq)} sequences at 128K context"
          f" (or {int(room/(L*(kvl+rope)*8192/GB))} at 8K)")
    print(f"  decode floor     : {ms:.0f} ms/token = {1000/ms:.1f} tok/s single stream")
    # batched throughput: bandwidth is shared, so aggregate is what matters
    print(f"  batched: the same {PER_TOKEN} GB/token serves a whole batch, so")
    print(f"           throughput scales until compute saturates (~{agg*1000/PER_TOKEN*8:.0f} tok/s"
          f" at batch 8, before FLOPs limit)")
