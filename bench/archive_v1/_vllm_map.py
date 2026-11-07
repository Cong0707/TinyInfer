print("=== vLLM kernel 家族 vs 我们现状 ===")
print()
families = [
    ("Marlin W4A16 (fp4 专家)", "mma.sync.m16n8k32.s4 + cp.async + 离线重打包", "标量 FFMA 62GB/s", "3.87x"),
    ("CUTLASS scaled_mm (fp8)", "cutlass 分组 GEMM，per-block scale", "tilelang 兜底", "?"),
    ("fused_moe", "路由+topk+permute+GEMM+unpermute 融合", "分开做，多次 launch", "?"),
    ("FlashMLA / MLA attention", "变长 MLA decode kernel", "tilelang", "?"),
    ("PagedAttention", "块表 KV，非连续", "连续 KV 窗口", "?"),
    ("fused_add_rms_norm", "残差+norm 融合", "单独 kernel", "?"),
    ("silu_and_mul", "激活融合", "已有 swiglu", "同"),
    ("rotary_embedding", "RoPE 融合", "单独做", "?"),
    ("MoE permute/unpermute", "专家重排", "index 操作", "?"),
    ("DeepGEMM / DeepEP", "DeepSeek 官方 EP 通信", "无", "?"),
]
print(f"{'家族':24s} {'vLLM 技术':44s} {'我们':18s} 收益")
print("-" * 100)
for a, b, c, d in families:
    print(f"{a:24s} {b:44s} {c:18s} {d}")
print()
print("=== 优先级（按我们当前的瓶颈）===")
print("  P0  Marlin: fp4 专家是我们的主瓶颈，标量 62 vs 纯读 212 GB/s")
print("  P1  fused_moe: 减少 launch + 避免 permute 往返")
print("  P2  fused_add_rms_norm: 40 层 x 2 次，省 80 次 launch")
print("  P3  FlashMLA: 长上下文时的 attention")
print("  P4  DeepEP: 多卡部署才需要")