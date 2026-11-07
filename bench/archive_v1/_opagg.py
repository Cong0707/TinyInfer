
# 按 DeepSeek-V4.1-Flash 的真实配置算每步调用次数
# 40 层，每层: 1 次 gate_topk、6 专家 GEMV x3 投影、1 次 clamp_swiglu、若干 rms_norm
L = 40
EXPERTS = 6     # topk
GEMMS_PER_EXPERT = 3   # w1 gate / w3 up / w2 down

# 实测（us）
old = {
    "rms_norm":       84.26,
    "clamp_swiglu":   60.75,
    "gate_topk":     160.11,
    "marlin":        582.60,   # 单手写 GEMV
}
new = {
    "rms_norm":       11.53,
    "clamp_swiglu":   13.70,
    "gate_topk":      15.78,
    "marlin":        143.30,
}
# 次数/step
counts = {
    "rms_norm":     L * 2,        # attn 前后各一次（保守）
    "gate_topk":    L,
    "clamp_swiglu": L * EXPERTS,
    "marlin":       L * EXPERTS * GEMMS_PER_EXPERT,
}
print(f"{'item':<16} {'count':>8} {'old ms':>10} {'new ms':>10} {'saved ms':>10}")
print("-"*60)
to = tn = 0.0
for k in counts:
    o = old[k]*counts[k]/1000
    n = new[k]*counts[k]/1000
    to += o; tn += n
    print(f"{k:<16} {counts[k]:>8} {o:>10.1f} {n:>10.1f} {o-n:>10.1f}")
print("-"*60)
print(f"{'TOTAL':<16} {'':>8} {to:>10.1f} {tn:>10.1f} {to-tn:>10.1f}")
print(f"\n算子时间: {to:.0f} ms -> {tn:.0f} ms")
print(f"若整步就是这些: 1/{to/1000:.2f} -> 1/{tn/1000:.2f} tok/s")
