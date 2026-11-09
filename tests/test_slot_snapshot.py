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
    # 槽位池不报命中率了（见 snapshot() 的 docstring）：它只知道「搬了多少次」，
    # 不知道「有多少个专家本来就够用」—— 后者要按专家数数，只有
    # ExpertTiers.need() 拿得到完整的 want 列表。两处各算一遍就是两个对不上的
    # 命中率，所以统一从 collect_observability 出。
    assert "hit_rate" not in s


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


def test_snapshot_does_not_invent_a_hit_rate():
    """槽位池只报它真的知道的数：搬了多少次（loads），不报命中率。

    原来这里有 hits 计数和 hit_rate，但 hits 从来没被自增过（恒为 0）——
    第 2 章把它删了，改为在 collect_observability 里按专家数统一算。
    这条测试锁住「快照不再凭空造一个命中率」这件事。
    """
    p = _pool(8)
    p.loads = 3
    s = p.snapshot()
    assert "hit_rate" not in s
    assert s["loads"] == 3


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
