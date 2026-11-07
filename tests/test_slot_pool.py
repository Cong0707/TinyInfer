"""显存槽位池的行为约束。

这一组测试锁住三件曾经出过事的事情：
  1. 池子满的时候不能卡死（覆盖式分配，不做专门的「释放」）；
  2. 槽位表必须和 _by_key 完全一致（不一致时 kernel 会静静读到别人的权重）；
  3. 已经驻留的专家不能被重新搬一遍（每步重读 4.5GB 就是 decode 掉到 1 tok/s 的根因）。
"""
import pytest

torch = pytest.importorskip("torch")

from tinyinfer.loading import VramSlotCache  # noqa: E402

BLOB = 1024


@pytest.fixture()
def pool():
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    cache = VramSlotCache(8, BLOB, device="cuda", dtype=torch.uint8,
                          pin_staging=True, slot_arena=True)
    cache._ensure()
    return cache


def _loader(layer, want):
    return {int(e): bytes([int(e) & 0xFF]) * BLOB for e in want}


def test_coverage_allocation_never_wedges(pool):
    """8 格池子轮转 20 个专家，40 步之后不能有拿不到槽位的步。"""
    for step in range(40):
        want = [(step * 5 + k) % 20 for k in range(6)]
        miss, placed = pool.ensure(0, want, _loader)
        assert placed == miss, f"step {step}: placed {placed} of {miss}"
        for e in want:
            idx = pool._by_key.get((0, e))
            assert idx is not None and 0 <= idx < pool.slots
        # 用完了才还（MoE 是在本层 forward 末尾对 _last_ids 做同样的事）
        for e in want:
            pool.release(0, e)


def test_table_matches_by_key(pool):
    """device 上的槽位表是 kernel 唯一的信息来源，必须和 Python 侧完全一致。"""
    tab = pool.slot_table(4, 64)
    for step in range(25):
        pool.ensure(0, [(step * 3 + k) % 18 for k in range(6)], _loader)
        pool.wait_ready(list(pool._by_key.values()))
    bad = [(key, idx, int(tab[key[0], key[1]]))
           for key, idx in pool._by_key.items() if int(tab[key[0], key[1]]) != idx]
    assert not bad, f"table disagrees with _by_key: {bad[:5]}"


def test_resident_experts_are_not_reloaded(pool):
    """同一个专家第二次进来必须命中，不能再搬一遍。"""
    want = [3, 7, 11, 19, 23, 31]
    pool.ensure(0, want, _loader)
    loads = pool.loads
    for _ in range(5):
        miss, placed = pool.ensure(0, want, _loader)
        assert (miss, placed) == (0, 0)
    assert pool.loads == loads, "already-resident experts were fetched again"


def test_release_keeps_the_slot_resident(pool):
    """release 只把引用还给池子，槽位和数据都还在（覆盖时才真的换人）。"""
    pool.ensure(0, [5, 6], _loader)
    for e in (5, 6):
        pool.release(0, e)
    assert pool._by_key[(0, 5)] is not None
    miss, placed = pool.ensure(0, [5, 6], _loader)
    assert (miss, placed) == (0, 0)


def test_no_slot_left_in_transit(pool):
    """一轮跑完不该有槽位卡在 reserved/copying 上。"""
    pool.ensure(0, list(range(6)), _loader)
    pool.wait_ready(list(pool._by_key.values()))
    stuck = [i for i in range(pool.slots)
             if pool._state[i] in (pool.RESERVED, pool.COPYING)]
    assert not stuck, f"slots stuck in transit: {stuck}"
