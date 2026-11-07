print("=== 我错在哪：把单流带宽除法当成了吞吐 ===")
print()
print("我的算法：每 token 读 10.96 GB / 19.2 TB/s = 571 ms -> 1.8 tok/s")
print()
print("错在这个 10.96 GB 是【batch 里每个 token 各自要读一份】吗？不是。")
print("权重读一次，服务整个 batch。所以 batch=32 时：")
print()
GB = 2**30
per_token_bytes = 10.96   # GB, weights a single token path touches
agg_bw = 19.2             # TB/s, 4 x H200
batch = 32
print(f"  batch={batch}: 权重读取仍是 {per_token_bytes:.2f} GB（不是 x{batch}）")
print(f"  但 FLOPs 是 {batch} 倍")
print()
# so at batch>1 the limit flips to compute
ACTIVE = 37e9
flops = 2 * ACTIVE * batch
fp8_tflops = 4 * 1979e12
print(f"  batch={batch} FLOPs/token: {flops/1e12:.1f} TFLOP")
print(f"  4 x H200 fp8 算力      : {fp8_tflops/1e12:.0f} TFLOP/s")
print(f"  -> 算力上限            : {fp8_tflops/flops:.0f} tok/s")
print()
print("8 x H200 (公开实测，671B R1, vLLM): 1000 tok/s @ batch 32")
print()
print("=== 单流 vs 吞吐，两个不同的数 ===")
print("  单流 decode  : 受 HBM 带宽约束，个位数到几十 tok/s")
print("  批量吞吐     : 受算力约束，几百到上千 tok/s")
print("  我之前把『单流的带宽下限』当成了『整机的吞吐』，所以算出 1.8。")
print()
print("=== 修正后的配置结论 ===")
print("  8 x H200 (141GB each = 1128 GB): 权重 286 GB + KV 余量 840 GB")
print("    单流 decode  : 20-40 tok/s")
print("    批量吞吐     : 800-1000 tok/s @ batch 32")
print("  4 x H200 (564 GB): 权重 286 GB + KV 余量 278 GB")
print("    单流 decode  : 10-20 tok/s")
print("    批量吞吐     : 400-500 tok/s")