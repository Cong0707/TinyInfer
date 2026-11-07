import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.loading import VramSlotCache

SLOTS, BLOB = 8, 1024
sl = VramSlotCache(SLOTS, BLOB, device="cuda", dtype=torch.uint8,
                   pin_staging=True, slot_arena=True)
sl._ensure()
tab = sl.slot_table(4, 64)
print("slots=%d free=%d" % (sl.slots, len(sl._free)), flush=True)

calls = {"n": 0}
def loader(layer, want):
    calls["n"] += 1
    return {e: bytes([e & 0xFF]) * BLOB for e in want}

for step in range(30):
    experts = [(step * 5 + k) % 20 for k in range(6)]      # 20 个专家轮转
    miss, placed = sl.ensure(0, experts, loader)
    sl.wait_ready([sl._by_key[(0, e)] for e in experts if (0, e) in sl._by_key])
    for e in experts:
        idx = sl._by_key.get((0, e))
        assert idx is not None, "not resident step=%d e=%d" % (step, e)
        assert int(tab[0, e]) == idx, "table mismatch step=%d e=%d" % (step, e)
    for e in experts:
        if (0, e) in sl._by_key:
            sl.release(0, e)

bad = sum(1 for (l, e), i in sl._by_key.items() if int(tab[l, e]) != i)
print("bykey=%d evict=%d loads=%d loader_calls=%d table_mismatch=%d" %
      (len(sl._by_key), sl.evictions, sl.loads, calls["n"], bad), flush=True)
assert bad == 0
stuck = [i for i in range(sl.slots) if sl._state[i] in (sl.RESERVED, sl.COPYING)]
print("stuck=%d" % len(stuck), flush=True)
assert not stuck
print("ENSURE_OK", flush=True)
