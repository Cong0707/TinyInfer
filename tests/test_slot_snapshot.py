"""槽位池的运维可见性：snapshot() 必须说真话。

线上出问题时（命中率突然掉、开始换出），光看一个 miss 数字分不清是
「路由变了」还是「池子被某个长会话占死了」。snapshot 把每个状态的数量、
命中率、哪些层占了多少槽位摊开。

这组测试锁住它的**自洽性**：状态计数之和等于容量、占用数等于映射大小、
命中率算得对。诊断工具说谎比没有更糟。
"""

import pytest

torch = pytest.importorskip("torch")

from tinyinfer.loading import VramSlotCache  # noqa: E402


def _pool(n=8, blob=64):
    return VramSlotCache(slots=n, blob_bytes=blob, device="cpu")


def test_empty_pool_snapshot():
    p = _pool(8)
    s = p.snapshot()
    assert s["slots"] == 8
    st = s["states"]
    assert st["free"] == 8, st
    assert sum(st.values()) == 8, st
    assert s["occupied"] == 0
    assert s["fill_ratio"] == 0.0
    assert s["hit_rate"] == 0.0


def test_states_always_sum_to_capacity():
    """任何时刻，五个状态的数量之和必须等于槽位数。"""
    p = _pool(8)
    for layer in range(3):
        for e in range(4):
            p.acquire(layer, e)
        s = p.snapshot()
        assert sum(s["states"].values()) == s["slots"], s


def test_occupied_matches_bookkeeping():
    p = _pool(8)
    for e in range(3):
        p.acquire(0, e)
    s = p.snapshot()
    assert s["occupied"] == len(p._by_key), (s["occupied"], len(p._by_key))
    assert s["occupied"] == 3
    assert s["fill_ratio"] == pytest.approx(3 / 8)


def test_hit_rate_is_computed_from_counters():
    """池子只记 hits 和 loads（每次真正搬一次算一个 load）。"""
    p = _pool(8)
    p.hits = 7
    p.loads = 3
    s = p.snapshot()
    assert s["hit_rate"] == pytest.approx(0.7)
    assert s["hits"] == 7 and s["loads"] == 3


def test_busiest_layers_reports_real_counts():
    p = _pool(8)
    for e in range(3):
        p.acquire(5, e)
    p.acquire(7, 0)
    s = p.snapshot()
    assert s["layers_present"] == 2
    top = dict(s["busiest_layers"])
    assert top.get(5) == 3, s["busiest_layers"]
    assert top.get(7) == 1
