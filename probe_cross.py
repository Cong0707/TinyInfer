"""两条约束的交叉点：KV 容量 vs 专家槽位。"""
import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.kv_tiers import plan_slots, batch_slots_bound, union_estimate

GB = 2**30
slots = 1359            # L20
n_layers = 40
topk = 6

print("=== L20 (1359 专家槽位) ===")
for ctx in (128_000, 1_048_576):
    by_kv = plan_slots(vram_free_bytes=44*GB - 4*GB, context_tokens=ctx)
    by_slots = batch_slots_bound(slots=slots, n_layers=n_layers, topk=topk)
    print("ctx=%7d: KV->%2d 条, 槽位->%2d 条, 实际=%d" % (
          ctx, by_kv, by_slots, min(by_kv, by_slots)), flush=True)

print()
print("=== 每并发数的专家并集 x 40 层 ===")
for n in range(1, 10):
    per = union_estimate(n)
    need = per * n_layers
    print("  %d 条: %.1f 专家/层 -> %4.0f 槽位 %s" % (
          n, per, need, "OK" if need <= slots else "超出"), flush=True)
