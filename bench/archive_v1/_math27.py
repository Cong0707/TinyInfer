"""The arithmetic that decides whether 27B can be fast on this box."""
import json
from pathlib import Path

d = json.load(open(r"E:\Models\Qwen3.8-27B\config.json", encoding="utf-8"))
tc = d.get("text_config", d)
H = tc["hidden_size"]; L = tc["num_hidden_layers"]; I = tc["intermediate_size"]
V = tc["vocab_size"]; nh = tc["num_attention_heads"]; nkv = tc["num_key_value_heads"]
hd = tc["head_dim"]; nk = tc["linear_num_key_heads"]; nv = tc["linear_num_value_heads"]
kd = tc["linear_key_head_dim"]; vd = tc["linear_value_head_dim"]
kinds = tc["layer_types"]

# per-layer weights in bf16
vdim = nv * vd
lin = (H * (nk*kd*2 + vdim + nv + nv) + vdim * H) * 2          # qkv,z,b,a + out
full = ((nh*hd*2 + nkv*hd*2 + nkv*hd*2) * H + H * nh * hd) * 2  # q(with gate),k,v,o
mlp = 3 * H * I * 2
per = {}
for k in ("linear_attention", "full_attention"):
    per[k] = mlp + (lin if k == "linear_attention" else full)
n_lin = kinds.count("linear_attention"); n_full = kinds.count("full_attention")
body = n_lin * per["linear_attention"] + n_full * per["full_attention"]
emb = V * H * 2
head = 0 if tc.get("tie_word_embeddings") else V * H * 2
total = body + emb + head
print(f"layers: {L} ({n_lin} linear @ {per['linear_attention']/2**20:.0f}MB, "
      f"{n_full} full @ {per['full_attention']/2**20:.0f}MB)")
print(f"body {body/2**30:.1f} GB + embed {emb/2**30:.2f} GB + head {head/2**30:.2f} GB "
      f"= {total/2**30:.1f} GB")
print()
VRAM, RAM = 8.0, 30.0     # usable
print(f"usable: VRAM {VRAM} GB, RAM {RAM} GB")
for bits in (16, 8, 4, 2):
    size = total * bits / 16
    where = ("all in VRAM" if size <= VRAM * 0.9 else
             "all in RAM" if size <= RAM else
             "must stream from disk")
    print(f"  {bits:2d}-bit: {size/2**30:5.1f} GB -> {where}")
print()
for rate, name in ((2.85, "measured seq read"), (4.5, "SSD peak"), (12.0, "PCIe H2D")):
    streamed = total / 2**30
    sec = streamed / rate
    print(f"stream {streamed:.1f} GB at {rate:5.1f} GB/s ({name}): {sec:5.1f}s/token "
          f"= {1/sec:.3f} tok/s")
