print("=" * 78)
print("DeepSeek-V4.1-Flash-1017 on H200: decode throughput")
print("=" * 78)
print()
# measured model constants
FLOPS_TOK = 0.027e12
BYTES_TOK = 9.24e9
L, TOPK = 40, 6
EXP1_B = 18.38e6 * 0.5     # fp4 bytes per expert

print(f"激活参数 13.5 B/token, FLOPs {FLOPS_TOK/1e12:.3f} TFLOP")
print(f"权重读取 {BYTES_TOK/1e9:.2f} GB/token (fp8 dense + fp4 experts + head)")
print(f"KV: 1.25 GB/序列 @1M, 0.156 GB @128K, 0.010 GB @8K")
print()

BW = 4.8e12; FP8 = 1979e12
for n in (2, 3):
    bw = n * BW; fl = n * FP8
    vram = n * 141.0
    kv_room = vram - 286.4
    print("=" * 78)
    print(f"{n} x H200: {bw/1e12:.1f} TB/s, {fl/1e12:.0f} TFLOP/s, {vram:.0f} GB VRAM")
    print(f"  weights 286.4 GB -> KV room {kv_room:.0f} GB")
    print("=" * 78)
    print()
    print("  单流上限（权重只服务 1 条）:")
    ideal = bw / BYTES_TOK * 0.55
    print(f"    {ideal:.0f} tok/s (55% 效率)")
    print()
    print("  并发曲线（8K 上下文）:")
    print("    conc   单条 tok/s   总 tok/s   瓶颈")
    kv8k = 1.25e9 * 8192 / 1048576
    for conc in (1, 2, 5, 10, 20, 50, 100, 200):
        step_b = BYTES_TOK + kv8k * conc
        bw_ms = step_b / bw * 1000
        fl_ms = FLOPS_TOK * conc / fl * 1000
        bound = max(bw_ms, fl_ms)
        per = 1000 / bound * 0.55
        who = "bandwidth" if bw_ms >= fl_ms else "compute"
        print(f"    {conc:5d} {per:11.0f} {per*conc:10.0f}   {who}")
    print()