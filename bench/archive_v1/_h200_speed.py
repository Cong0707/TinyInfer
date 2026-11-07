"""H200 deployment sizing for DeepSeek-V4.1-Flash-1017, from the measured checkpoint."""
GB = 2**30

# ---- measured from the checkpoint ----------------------------------------
W_TOTAL = 475.24          # GB on disk, fp8 non-expert + fp4 experts
W_EXPERT = 275.67         # GB, all 40*384 blobs
W_LAYER = 194.22          # GB, attention/MLA/norms per layer
W_TABLES = 3.94           # embedding + head + norm
W_SHARED = 1.42

L, E, TOPK = 40, 384, 6
PER_EXPERT_MB = 18.38

# ---- H200 ----------------------------------------------------------------
HBM = 141.0               # GB HBM3e per card
BW = 4.8                  # TB/s per card
print(f"checkpoint: {W_TOTAL:.0f} GB (fp8 body + fp4 experts)")
print(f"per card   : H200 {HBM:.0f} GB @ {BW:.1f} TB/s")
print()

# ---- weights: how many cards just to hold them ---------------------------
print("=== weights only ===")
for n in (2, 4, 8):
    cap = n * HBM
    left = cap - W_TOTAL
    print(f"  {n} x H200 = {cap:5.0f} GB -> {left:+6.0f} GB left for KV + activations"
          f"{'  TOO TIGHT' if left < 40 else ''}")
print()

# ---- per token: what has to move -----------------------------------------
# MoE: each token reads topk experts per layer; those are 18.38MB each
experts_per_token = L * TOPK
bytes_experts = experts_per_token * PER_EXPERT_MB * 2**20
bytes_layer = W_LAYER * GB          # every layer's attention runs for every token
bytes_tables = W_TABLES * GB
per_token = bytes_experts + bytes_layer + bytes_tables
print("=== bytes read per token (weights) ===")
print(f"  attention/MLA path : {bytes_layer/GB:6.1f} GB  (all 40 layers, dense)")
print(f"  routed experts     : {experts_per_token} x {PER_EXPERT_MB}MB = "
      f"{bytes_experts/GB:6.1f} GB")
print(f"  embedding + head   : {bytes_tables/GB:6.1f} GB")
print(f"  TOTAL              : {per_token/GB:6.1f} GB per token")
print()

# ---- if fully resident: decode is bandwidth bound ------------------------
# with all weights in HBM, a token streams what it uses, not the whole model
print("=== decode speed with all weights in HBM ===")
for n in (4, 8):
    agg = n * BW                             # TB/s, weights are sharded so bandwidth adds
    per_tok_s = per_token / GB / agg         # seconds if perfectly bandwidth bound
    tps = 1 / per_tok_s
    print(f"  {n} x H200: aggregate {agg:.1f} TB/s -> "
          f"{per_token/GB:.1f} GB/token / {agg:.1f} TB/s = {per_tok_s*1000:.1f} ms/token")
    print(f"             = {tps:.0f} tok/s for one sequence")
print()
print("  (single-sequence decode is latency bound, not throughput bound: the")
print("   arithmetic above is the floor, real kernels land at 50-80% of it)")
