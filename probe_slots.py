import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.loading import VramSlotCache

SLOTS, BLOB = 8, 1024
sl = VramSlotCache(SLOTS, BLOB, device="cuda", dtype=torch.uint8,
                   pin_staging=True, slot_arena=True)
sl._ensure()
print("slots=%d free=%d" % (sl.slots, len(sl._free)), flush=True)
tab = sl.slot_table(4, 64)

ok = 0
for step in range(40):
    req = [(0, (step * 7 + k) % 24) for k in range(6)]   # 24 个专家轮转，池子只有 8 格
    for layer, e in req:
        idx, ready = sl.acquire(layer, e)
        assert idx >= 0, "acquire failed step=%d e=%d state=%s" % (step, e, sl._state)
        assert 0 <= idx < sl.slots
        if not ready:
            sl.copy_in(idx, b"" * BLOB)
            sl.publish(idx)
        assert sl._by_key[(layer, e)] == idx
    for layer, e in req:
        sl.release(layer, e)
    if step % 10 == 0:
        print("step=%d evict=%d loads=%d hits=%d bykey=%d free=%d" %
              (step, sl.evictions, sl.loads, sl.hits, len(sl._by_key), len(sl._free)), flush=True)

# 表一致性：在住的专家，表里必须等于 _by_key；不在住的必须是 -1
bad = 0
for (layer, e), idx in sl._by_key.items():
    v = int(tab[layer, e])
    if v != idx:
        bad += 1
        print("TABLE MISMATCH", layer, e, "table", v, "bykey", idx)
print("table_mismatch=%d evict=%d loads=%d" % (bad, sl.evictions, sl.loads), flush=True)
assert bad == 0

# 状态机：所有槽位都应处于 FULL 族（staged/in_use），不应残留 copying/reserved
res = [i for i in range(sl.slots) if sl._state[i] in (sl.RESERVED, sl.COPYING)]
print("stuck_slots=%d" % len(res), flush=True)
assert not res
print("SLOT_POOL_OK", flush=True)
