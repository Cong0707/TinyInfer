"""会话被换下槽位时的历史保留：park / resume。

KVPool.release 表示「这条会话结束了」，历史可以丢；park 表示「它还会回来」，
历史必须留着 —— 1M 上下文重新 prefill 是几分钟的重复计算，这就是
「并发」和「长上下文」能不能同时成立的差别。
"""

import pytest

from tinyinfer.sequence import KVPool, Sequence


def test_park_keeps_history_and_resume_restores_it():
    pool = KVPool(1)
    seq = Sequence(prompt=[1, 2, 3])
    assert pool.acquire(seq) == 0
    seq.state["ckv"] = "compressed-kv-bytes"
    pool.park(seq)
    assert pool.free == 1, "the slot must go back to the pool"
    assert pool.parked == 1
    # 回来：历史必须原样装回去
    assert pool.acquire(seq) is not None
    assert seq.state.get("ckv") == "compressed-kv-bytes"
    assert pool.restored == 1


def test_release_still_drops_history():
    """会话真的结束了才丢历史：release 的语义不能被 park 改掉。"""
    pool = KVPool(1)
    seq = Sequence(prompt=[1])
    pool.acquire(seq)
    seq.state["ckv"] = "x"
    pool.release(seq)
    again = Sequence(prompt=[2])
    pool.acquire(again)
    # 新会话拿到同一个槽位，不能带着别人的上下文
    assert "ckv" not in again.state


def test_park_returns_the_bytes_saved():
    torch = pytest.importorskip("torch")
    pool = KVPool(1)
    seq = Sequence(prompt=[1])
    pool.acquire(seq)
    seq.state["kv"] = torch.zeros(1024, dtype=torch.bfloat16)
    n = pool.park(seq)
    assert n == 2048, n


def test_multiple_parked_sessions_fit_one_slot():
    """一条槽位轮流服务多条会话：这就是「卡上只放活跃的」。"""
    pool = KVPool(1)
    seqs = [Sequence(prompt=[i]) for i in range(4)]
    for i, s in enumerate(seqs):
        pool.acquire(s)
        s.state["ckv"] = f"hist-{i}"
        pool.park(s)
    assert pool.parked == 4
    # 轮流回来，各自拿到自己的历史
    for i, s in enumerate(seqs):
        pool.acquire(s)
        assert s.state.get("ckv") == f"hist-{i}"
        pool.park(s)


def test_parked_sessions_are_bounded():
    """内存不是无限的：超过 max_parked 就丢最老的，不能无限涨。"""
    pool = KVPool(1, max_parked=2)
    seqs = [Sequence(prompt=[i]) for i in range(5)]
    for i, s in enumerate(seqs):
        pool.acquire(s)
        s.state["ckv"] = f"h{i}"
        pool.park(s)
    assert len(pool._parked) == 2, pool._parked
    assert pool.dropped == 3
    assert pool.resume(seqs[-1])
    assert pool.resume(seqs[-2])

