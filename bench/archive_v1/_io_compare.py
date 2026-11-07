"""Every token's real I/O, side by side. This is what decides tok/s."""
GB = 2**30; MB = 2**20

# ---- DeepSeek: what one token actually reads -----------------------------
# From the engine's own startup log (measured, not guessed):
#   layers=10/40 stream=30 buf=3 cache=14GB layer=178MB expert=17.9MB
DS_LAYER_MB = 178.0        # per-layer non-expert weights, fp8
DS_EXPERT_MB = 17.9        # one routed expert blob, fp4
DS_STREAM_LAYERS = 30
DS_TOP_K = 6
DS_HIT = 0.88              # converged expert-cache hit rate (measured over 7 passes)

ds_layer_bytes = DS_STREAM_LAYERS * DS_LAYER_MB * MB
ds_expert_bytes = DS_STREAM_LAYERS * DS_TOP_K * (1 - DS_HIT) * DS_EXPERT_MB * MB
ds_total = ds_layer_bytes + ds_expert_bytes
print("=== DeepSeek-V4.1-Flash (40 layers, 384 experts, top-6) ===")
print(f"  streamed layer weights : {DS_STREAM_LAYERS} x {DS_LAYER_MB}MB = "
      f"{ds_layer_bytes/GB:.2f} GB")
print(f"  expert misses          : {DS_STREAM_LAYERS} x {DS_TOP_K} x "
      f"{(1-DS_HIT)*100:.0f}% x {DS_EXPERT_MB}MB = {ds_expert_bytes/GB:.2f} GB")
print(f"  TOTAL per token        : {ds_total/GB:.2f} GB")
print(f"  at 2.85 GB/s disk      : {ds_total/GB/2.85:.2f} s/token -> "
      f"{1/(ds_total/GB/2.85):.2f} tok/s")
print(f"  measured               : 0.519 s/token -> 1.93 tok/s")
print()

# ---- Qwen 27B: dense, so every parameter every token ---------------------
Q_TOTAL = 47.6
print("=== Qwen3.8-27B (64 layers, DENSE) ===")
print(f"  every layer, every token: {Q_TOTAL:.1f} GB")
print(f"  at 2.85 GB/s disk       : {Q_TOTAL/2.85:.1f} s/token -> "
      f"{1/(Q_TOTAL/2.85):.3f} tok/s")
print()
print(f"ratio: DeepSeek moves {ds_total/GB:.2f} GB per token, Qwen27B moves "
      f"{Q_TOTAL:.1f} GB -> {Q_TOTAL/(ds_total/GB):.0f}x more")
print()
print("Why: DeepSeek is 600B parameters but only 1.56% of its experts fire per token,")
print("and the expert cache absorbs 88% of those. A dense 27B has no such discount:")
print("all 64 layers run for all 27B, so all 47.6GB is live every step.")
