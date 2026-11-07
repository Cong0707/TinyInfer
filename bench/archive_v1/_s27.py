import json
d = json.load(open(r"E:\Models\Qwen3.8-27B\config.json", encoding="utf-8"))
tc = d.get("text_config", d)
H, L = tc["hidden_size"], tc["num_hidden_layers"]
I = tc["intermediate_size"]
nh, nkv, hd = tc["num_attention_heads"], tc["num_key_value_heads"], tc["head_dim"]
lt = tc["layer_types"]
n_full = sum(1 for t in lt if t == "full_attention")
n_lin = L - n_full
# per-layer weight bytes (bf16 = 2)
# full attention: q(24*256) k(4*256) v(4*256) o(H) but q_proj has an output gate -> x2 for q
full = (nh*hd*2 + nkv*hd*2 + nkv*hd*2 + H*H)*2 + (nh*hd + nkv*hd*2)*2  # rough
# linear attn (gated delta): in_proj_qkv, z, b, a, out_proj, conv, norms
vd = 4 * 128  # guess: num_v_heads * value_head_dim
lin = (H*(2*4*128 + vd + 4 + 4) + vd*H)*2
mlp = 3 * H * I * 2
print(f"layers: {L} ({n_full} full, {n_lin} linear)")
print(f"  MLP per layer: {mlp/2**20:.0f} MB")
print(f"  full attn  ~: {full/2**20:.0f} MB")
print(f"  linear attn~: {lin/2**20:.0f} MB")
total = n_full*(full+mlp) + n_lin*(lin+mlp)
print(f"  => transformer body: {total/2**30:.1f} GB")
emb = tc["vocab_size"]*H*2
print(f"  embeddings: {emb/2**30:.2f} GB (untied)")
print(f"  => total ~{(total+emb)/2**30:.1f} GB")
# KV cost
kv_per_pos = n_full*2*nkv*hd*2
print(f"KV per position: {kv_per_pos/1024:.0f} KB -> 4K ctx = {4096*kv_per_pos/2**30:.2f} GB")
