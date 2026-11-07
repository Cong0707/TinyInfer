"""分级管理器（显存 / 内存 / 磁盘）的行为约束。

MoE 现在只做两件事：说「我这一层要谁」，然后等 DMA。所有分级决策在这里。
这一组测试用假的 store 覆盖三种来源，确保：

  1. 显存命中时不去碰内存和磁盘（decode 稳态的热点路径，多一次查询就是
     每 token 240 次无谓开销）；
  2. 内存命中时不读盘；
  3. 内存没有时才读盘，并且读到的字节会进内存（下次不用再读）；
  4. 已经驻留的专家不会被重复搬。
"""

import pytest

torch = pytest.importorskip("torch")

from tinyinfer.expert_tiers import ExpertTierManager  # noqa: E402
from tinyinfer.loading import VramSlotCache  # noqa: E402

BLOB = 1024


class FakeRam:
    def __init__(self):
        self.entries = {}
        self.lookups = 0

    def get(self, layer, expert, blocking=True, admit=True):
        self.lookups += 1
        return self.entries.get((layer, expert))

    def contains(self, layer, expert):
        return (layer, expert) in self.entries

    def admit(self, layer, expert, data):
        self.entries[(layer, expert)] = data
        return data

    def prefetch(self, layer, experts):
        raise AssertionError("warm-up must not run in these tests")


class FakeIndex:
    def expert_blob(self, layer, expert):
        return (layer, expert)


class FakeReader:
    def __init__(self):
        self.reads = []

    def read_many(self, blobs):
        self.reads.extend(blobs)
        return [bytes([b[1] & 0xFF]) * BLOB for b in blobs]


class FakeStore:
    def __init__(self, slots):
        self.slots = slots
        self.ram = FakeRam()
        self.index = FakeIndex()
        self.reader = FakeReader()


@pytest.fixture()
def tier():
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    slots = VramSlotCache(16, BLOB, device="cuda", dtype=torch.uint8,
                          pin_staging=True, slot_arena=True)
    slots._ensure()
    store = FakeStore(slots)
    return store, ExpertTierManager(store, n_layers=4, n_experts=64, warm=False)


def test_vram_hit_skips_ram_and_disk(tier):
    store, mgr = tier
    mgr.need(0, [1, 2, 3])
    before = store.ram.lookups
    reads = len(store.reader.reads)
    assert mgr.need(0, [1, 2, 3]) == 0
    assert store.ram.lookups == before, "a resident expert must not consult RAM"
    assert len(store.reader.reads) == reads, "a resident expert must not read disk"


def test_ram_hit_skips_disk(tier):
    store, mgr = tier
    store.ram.entries[(1, 5)] = bytes([5]) * BLOB
    reads = len(store.reader.reads)
    mgr.need(1, [5])
    assert len(store.reader.reads) == reads, "a RAM hit must not go to disk"
    assert mgr.resident(1, 5)


def test_disk_read_admits_into_ram(tier):
    store, mgr = tier
    mgr.need(2, [7])
    assert len(store.reader.reads) == 1
    assert store.ram.contains(2, 7), "bytes read from disk must be kept in RAM"


def test_disk_read_happens_once(tier):
    store, mgr = tier
    mgr.need(3, [9])
    reads = len(store.reader.reads)
    mgr.release(3, [9])
    # 换一个池子把这一格挤掉，再从内存命中
    store.slots.ensure(3, list(range(20, 36)),
                       lambda lay, want: {e: bytes([e]) * BLOB for e in want})
    mgr.release(3, list(range(20, 36)))
    mgr.need(3, [9])
    assert len(store.reader.reads) == reads, "RAM should have served the second use"


def test_stats_track_the_tiers(tier):
    store, mgr = tier
    mgr.need(0, [1, 2])
    mgr.need(0, [1])
    assert mgr.stats["disk"] + mgr.stats["ram_hit"] >= 2
    assert mgr.stats["vram_hit"] >= 1
