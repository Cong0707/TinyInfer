GB = 2**30
print("=== 我错在哪 ===")
print()
print("错法：把每 token 读的权重字节 / 聚合带宽 = 每 token 时间")
print("  10.96 GB / 19.2 TB/s = 571 ms -> 1.8 tok/s")
print()
print("这个算法的两个致命错误：")
print()
print("1) 权重不需要每 token 从 HBM 重读。H100/H200 有 50-60 MB L2，")
print("   而 MLA 的每层权重只有 138 MB -- 单层权重在 L2 里有大量命中。")
print("   真正必须走 HBM 的是【每 token 不同的数据】：KV cache 和激活。")
print()
print("2) 专家权重只有命中的那部分要读，而 w1/w3/w2 是稀疏 gather，")
print("   DeepEP/DeepGEMM 这类 kernel 就是为此写的，效率远高于暴力 GEMV。")
print()
print("=== 正确的算法：算力 + KV 带宽 ===")
print()
# FLOPs per token for a 600B-class MoE with 37B active params
ACTIVE = 37e9   # active params per token (671B model, ~5.5% activated)
flops_per_token = 2 * ACTIVE
print(f"激活参数/token      : {ACTIVE/1e9:.0f} B")
print(f"FLOPs/token         : 2 x {ACTIVE/1e9:.0f}B = {flops_per_token/1e12:.1f} TFLOP")
print()
# H200 bf16 dense: 989 TFLOPS (with sparsity 1979), fp8: 1979 TFLOPS
for name, tflops in (("bf16", 989e12), ("fp8", 1979e12)):
    tps = tflops / flops_per_token
    print(f"  H200 {name}: {tflops/1e12:.0f} TFLOP/s -> {tps:.0f} tok/s (算力上限)")
print()
# 8 cards: FLOPs scale, but weights must be split so there is cross-card traffic
for n in (4, 8):
    tps_1 = 1979e12 / flops_per_token
    print(f"  {n} x H200 fp8 算力上限: {n*tps_1:.0f} tok/s (batch>1)")
    print(f"      含通信与效率损失 (50%): {n*tps_1*0.5:.0f} tok/s")