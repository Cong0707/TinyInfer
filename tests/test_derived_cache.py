"""容量驱动的反量化缓存：小卡转得快，大卡一次装下。

这个模块要锁住一件事：wo_a 这类「重算很贵但可重算」的派生结果，**不该**按
层数写死缓存多少份。容量由芯片上真实剩多少显存决定，满了按 LRU 覆盖。
"""

import pytest

torch = pytest.importorskip("torch")

from tinyinfer.derived_cache import DerivedCache  # noqa: E402


def _t(nbytes: int):
    return torch.zeros(nbytes // 2, dtype=torch.bfloat16, device="cuda")


def test_hit_and_miss_counting():
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    c = DerivedCache(64 << 20)
    calls = {"n": 0}

    def factory():
        calls["n"] += 1
        return _t(16 << 20)

    a = c.get_or_compute(0, "wo_a", factory)
    b = c.get_or_compute(0, "wo_a", factory)
    assert calls["n"] == 1, "the second call must hit"
    assert a is b
    assert c.hits == 1 and c.misses == 1


def test_capacity_evicts_lru():
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    # 容量 48MB，每份 16MB -> 最多 3 份
    c = DerivedCache(48 << 20)
    for layer in range(4):
        c.get_or_compute(layer, "wo_a", lambda: _t(16 << 20))
    assert len(c._items) <= 3, c.stats()
    assert c.evictions >= 1
    # 最近用过的还在，最早那份被顶掉
    assert c.get(0, "wo_a") is None
    assert c.get(3, "wo_a") is not None


def test_evicted_entry_is_recomputed_not_corrupted():
    """覆盖只是丢引用：下次重算，绝不会读到别人的数据。"""
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    c = DerivedCache(16 << 20)
    v0 = c.get_or_compute(0, "x", lambda: torch.full((8,), 1.0, device="cuda"))
    c.get_or_compute(1, "x", lambda: torch.full((8 << 19,), 2.0, device="cuda"))
    v0b = c.get_or_compute(0, "x", lambda: torch.full((8,), 1.0, device="cuda"))
    assert torch.equal(v0b, torch.full((8,), 1.0, device="cuda"))
    assert float(v0b[0]) == 1.0


def test_big_capacity_never_evicts():
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    c = DerivedCache(1 << 30)
    for layer in range(40):
        c.get_or_compute(layer, "wo_a", lambda: _t(1 << 20))
    assert c.evictions == 0
    assert len(c._items) == 40
    for layer in range(40):
        assert c.get(layer, "wo_a") is not None
