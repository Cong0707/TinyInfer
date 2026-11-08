"""1M 上下文的 KV 容量核算（用真实配置）。"""
import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.kv_tiers import plan_slots, KvTierPlan

GB = 2**30

# L20 的真实参数
vram_free = 44 * GB - 4 * GB      # 减去层权重和缓冲
head_dim = 512

for ctx in (128_000, 512_000, 1_048_576):
    n = plan_slots(vram_free_bytes=vram_free, context_tokens=ctx)
    print("L20 ctx=%7d -> sessions=%d" % (ctx, n), flush=True)

print()
for vram in (24, 44, 80):
    for ctx in (128_000, 1_048_576):
        n = plan_slots(vram_free_bytes=vram * GB - 4 * GB, context_tokens=ctx)
        print("VRAM=%2dGB ctx=%7d -> sessions=%d" % (vram, ctx, n), flush=True)
