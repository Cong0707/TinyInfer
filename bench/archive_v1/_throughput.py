print("=== H200 x N 的 decode 吞吐曲线 ===")
print()
FLOPS_TOK = 0.027e12      # TFLOP -> FLOP per token
BYTES_TOK = 9.24e9        # bytes per token (weights, all from HBM)
print(f"每 token: {FLOPS_TOK/1e12:.3f} TFLOP, {BYTES_TOK/1e9:.2f} GB 权重读取")
print()
# H200: 989 TFLOPS bf16 dense, 1979 fp8, 4.8 TB/s
FLOPS_CARD = 989e12       # conservative: fp8 weight x fp8 act -> use dense fp8 1979?
FLOPS_CARD_FP8 = 1979e12
BW_CARD = 4.8e12
print("H200 单卡: 989 TFLOP/s (bf16 dense), 1979 TFLOP/s (fp8), 4.8 TB/s")
print()
for n in (2, 3):
    bw = n * BW_CARD
    flops = n * FLOPS_CARD_FP8
    print(f"=== {n} 卡: {bw/1e12:.1f} TB/s, {flops/1e12:.0f} TFLOP/s ===")
    print()
    print("  单流（batch 1）：受带宽限制，权重只服务 1 个 token")
    ms = BYTES_TOK / bw * 1000
    print(f"    带宽下限 {ms:.2f} ms/token -> {1000/ms:.0f} tok/s")
    print()
    print("  大 batch：权重共享，瓶颈转向算力")
    print("    conc   每 step 读取    每 step FLOPs   时间约束     单条 tok/s   总 tok/s")
    for conc in (1, 8, 16, 32, 50, 100, 200):
        # bandwidth: weights once + KV per seq (small at 8K ctx)
        kv_per_seq = 1.25e9 * 8192 / 1048576   # KV at 8K ctx
        step_bytes = BYTES_TOK + kv_per_seq * conc
        bw_ms = step_bytes / bw * 1000
        # compute
        step_flops = FLOPS_TOK * conc
        cc_ms = step_flops / flops * 1000
        bound = max(bw_ms, cc_ms)
        per = 1000 / bound
        print(f"    {conc:5d} {step_bytes/1e9:11.2f} GB {step_flops/1e12:13.2f} TF",
              f"{bound:9.2f} ms {per:10.1f} {per*conc:9.0f}")
    print()