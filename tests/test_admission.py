"""调度器的准入：槽位满时换出旧会话，而不是让新请求干等。

实测发现的问题：ServingEngine 的文档一直写着「KVPool decides how many can
share a step」，但类里从来没有 pool —— 只有「len(running) >= max_batch」的
条数上限。所以上一轮加的 KVPool.park/resume 在生产路径上一次都没被调用过。

这一组锁住：
  1. 槽位满时先 park 一个旧会话（KV 沉到内存），给新请求让位；
  2. 被 park 的会话带着历史回到等待队列；
  3. 会话跑完把槽位还回池子；
  4. 没有可换的会话时，新请求仍然等待（不无限循环）。
"""

import pytest

from tinyinfer.sched import ServingEngine
from tinyinfer.sequence import Sequence


class _FakeEngine:
    """只够 ServingEngine 构造用；step 不会被调到这里。"""
    device = "cpu"
    pad_id = 0


def _seq(n=1):
    return Sequence(prompt=[1] * n, max_new_tokens=8)


def _push(eng, seq):
    eng._inbox.put(seq)


def test_serving_engine_owns_a_pool():
    """文档承诺的东西必须真的存在。"""
    eng = ServingEngine(_FakeEngine(), max_batch=3)
    assert hasattr(eng, "pool")
    assert eng.pool.capacity == 3


def test_admit_parks_instead_of_blocking():
    eng = ServingEngine(_FakeEngine(), max_batch=1)
    a = _seq()
    _push(eng, a)
    eng._admit()
    assert len(eng._running) == 1
    assert eng.pool.free == 0

    # 必须有历史才谈得上「换出」：park 只保存 state，空 state 不进 _parked
    # （那是正确的：没有 KV 可留，直接丢掉槽位即可）。
    eng._running[0].state["ckv"] = "history"

    b = _seq()
    _push(eng, b)
    eng._admit()
    assert eng.pool.parked >= 1, "槽位满时必须换出，而不是干等"
    assert len(eng._running) == 1
    assert eng._running[0].id == b.id, "新请求必须拿到槽位"


def test_parked_session_keeps_its_history():
    eng = ServingEngine(_FakeEngine(), max_batch=1)
    a = _seq()
    _push(eng, a)
    eng._admit()
    a.state["ckv"] = "history"
    _push(eng, _seq())
    eng._admit()
    assert a.id in eng.pool._parked
    assert eng.pool._parked[a.id].get("ckv") == "history"


def test_make_room_refuses_when_nothing_to_park():
    eng = ServingEngine(_FakeEngine(), max_batch=2)
    assert eng._make_room() is False


def test_finished_session_returns_its_slot():
    eng = ServingEngine(_FakeEngine(), max_batch=1)
    a = _seq()
    _push(eng, a)
    eng._admit()
    assert eng.pool.free == 0
    eng.pool.release(a)
    assert eng.pool.free == 1
