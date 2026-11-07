print("=== 严格单位 ===")
# 11.14 GB per token
bytes_per_token = 11.14 * 1024**3          # bytes
print(f"per token: {bytes_per_token/1024**3:.2f} GB = {bytes_per_token:.3e} bytes")
print()
for n in (1, 2, 3, 4, 8):
    bw_tbs = n * 4.8                        # TB/s
    bw_bps = bw_tbs * 1e12                  # bytes/s
    sec = bytes_per_token / bw_bps
    ms = sec * 1000
    print(f"{n} x H200: {bw_tbs:4.1f} TB/s -> {ms:7.3f} ms/token -> {1/sec:8.0f} tok/s")
print()
print("=== 对一下公开数据 ===")
print("8 x H200 理论上 290 ms/token 单流；公开说 1000 tok/s 是 batch 32 总吞吐")
print("  batch 32: 每 step 处理 32 token，权重共享 -> 单 step 仍 ~290 ms？")
print("  那 batch32 总吞吐 = 32 / 0.290 = 110 tok/s，不是 1000。")
print()
print("  差 9 倍，说明公开的 1000 tok/s 上下文更短（2K 而非全量），")
print("  且 batch32 时 KV 很小、权重命中 L2 更多。")