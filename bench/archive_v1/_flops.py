import json
from pathlib import Path
GB = 2**30; MB = 2**20

tc = json.load(open(r"E:\Models\DeepSeek-V4.1-Flash-1017\config.json",
                    encoding="utf-8"))["text_config"]
L = tc["num_hidden_layers"]; H = tc["hidden_size"]; E = tc["n_routed_experts"]
TOPK = tc["num_experts_per_tok"]; MI = tc["moe_intermediate_size"]
SH = tc["n_shared_experts"]
nh = tc["num_attention_heads"]; hd = tc["head_dim"]
ql = tc["q_lora_rank"]; ol = tc["o_lora_rank"]
V = tc["vocab_size"]
print(f"L={L} H={H} E={E} top{TOPK} moe_inter={MI} shared={SH}")
print(f"heads={nh} head_dim={hd} q_lora={ql} o_lora={ol} vocab={V}")

# --- active params per token ---
# routed experts: top-k x (3 matrices) each [H, MI]
per_expert = 3 * H * MI
routed = TOPK * per_expert
# shared expert
shared = SH * 3 * H * MI
# attention per layer: MLA
attn = ql*H + nh*hd*ql + (hd+64)*ql + nh*hd*ol + ol*H + nh*hd  # rough MLA params
gate = H * E
tot_params_per_layer = attn + gate + routed + shared
active = L * tot_params_per_layer
print()
print(f"per layer active params: {tot_params_per_layer/1e6:.1f} M")
print(f"  attention ~{attn/1e6:.1f}M, gate {gate/1e6:.1f}M, routed {routed/1e6:.1f}M, shared {shared/1e6:.1f}M")
print(f"total active per token : {active/1e9:.1f} B")
print(f"FLOPs per token (2x)   : {2*active/1e12:.3f} TFLOP")

# --- bytes per token from HBM (weights) ---
# fp8 for non-expert, fp4 for experts
dense_bytes = (attn + gate + shared) * L * 1       # fp8 = 1 byte
expert_bytes = TOPK * L * per_expert * 0.5          # fp4 = 0.5 byte
head_bytes = V * H * 1                              # head gemv, fp8
print()
print(f"dense bytes/token  : {dense_bytes/GB:.2f} GB")
print(f"expert bytes/token : {expert_bytes/GB:.2f} GB")
print(f"head bytes/token   : {head_bytes/GB:.2f} GB")
total_bytes = dense_bytes + expert_bytes + head_bytes
print(f"TOTAL bytes/token  : {total_bytes/GB:.2f} GB")
