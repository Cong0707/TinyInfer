"""调度器的端到端行为：多条会话共用一个/几个槽位轮流推进。

前面的 test_admission 测的是准入决策（park / release），这里测**真的跑起来**：
用假引擎（不碰 GPU）驱动 ServingEngine 的 step 循环，验证

  - 两条会话能同时推进（槽位够）
  - 槽位只有 1 个时，第二条不会饿死（被 park 后能回来）
  - 每条会话各自收齐自己的 token，不串味
"""

import pytest

torch = pytest.importorskip("torch")

from tinyinfer.sched import ServingEngine
from tinyinfer.sequence import Sequence


class _FakeEngine:
    """最小可用的引擎替身：forward 返回一个 (rows, 1, vocab) 的 logits。

    用 token 值本身决定下一个 token，这样能验证「每个会话拿到的是自己的
    序列」—— 串味时 token 会跳。
    """
    device = "cpu"
    pad_id = 0

    def __init__(self):
        self.calls = 0
        self.seen_rows = []

    def forward(self, ids, start_pos=0, positions=None):
        self.calls += 1
        self.seen_rows.append(int(ids.shape[0]))
        n = int(ids.shape[0])
        vocab = 64
        out = torch.zeros(n, ids.shape[1], vocab)
        for r in range(n):
            nxt = (int(ids[r, -1]) + 1) % vocab
            out[r, -1, nxt] = 10.0
        return out


def _drain(seq, limit=100):
    out = []
    q = seq.queue
    while True:
        item = q.get(timeout=1.0)
        if item is None:
            break
        if isinstance(item, BaseException):
            raise item
        out.append(int(item))
        if len(out) >= limit:
            break
    return out


def test_two_sessions_run_together_and_keep_their_own_tokens():
    eng = ServingEngine(_FakeEngine(), max_batch=2, idle_sleep=0.001)
    a = Sequence(prompt=[1], max_new_tokens=3)
    b = Sequence(prompt=[20], max_new_tokens=3)
    eng.start()
    try:
        eng.submit(a)
        eng.submit(b)
        ta = _drain(a)
        tb = _drain(b)
    finally:
        eng.stop()
    assert len(ta) == 3 and len(tb) == 3, (ta, tb)
    # 假引擎的规则：下一个 token = 上一个 + 1。a 从 1、b 从 20。
    assert ta == [2, 3, 4], ta
    assert tb == [21, 22, 23], tb


def test_one_slot_does_not_starve_the_second_session():
    """"槽位只有 1 个时，后来的会话也必须跑完（被 park 后能回来）。"""
    eng = ServingEngine(_FakeEngine(), max_batch=1, idle_sleep=0.001)
    a = Sequence(prompt=[1], max_new_tokens=3)
    b = Sequence(prompt=[20], max_new_tokens=3)
    eng.start()
    try:
        eng.submit(a)
        eng.submit(b)
        ta = _drain(a)
        tb = _drain(b)
    finally:
        eng.stop()
    assert len(ta) == 3, ta
    assert len(tb) == 3, tb
    assert ta == [2, 3, 4], ta
    assert tb == [21, 22, 23], tb

def test_parked_session_resumes_at_the_right_position():
    """被 park 又回来的会话，下次取的必须是「最后一个已产 token」的位置。

    以前的 bug：重新接纳时仍被标成 fresh，于是每轮重放 prompt、
    永远吐同一个 token（单槽位实测 [2,2,2]）。这条直接盯住那条不变量。
    """
    eng = ServingEngine(_FakeEngine(), max_batch=1, idle_sleep=0.001)
    a = Sequence(prompt=[1, 2], max_new_tokens=4)
    eng.start()
    try:
        eng.submit(a)
        ta = _drain(a)
    finally:
        eng.stop()
    # prompt 最后一个是 2，所以序列是 3,4,5,6
    assert ta == [3, 4, 5, 6], ta


def test_batch_width_never_exceeds_the_pool():
    """running 的长度不能超过槽位数 —— 否则 forward 的 batch 维就会超。"""
    eng = ServingEngine(_FakeEngine(), max_batch=2, idle_sleep=0.001)
    seqs = [Sequence(prompt=[i], max_new_tokens=2) for i in range(1, 6)]
    eng.start()
    try:
        for s in seqs:
            eng.submit(s)
        outs = [_drain(s, limit=5) for s in seqs]
        assert len(eng._running) <= 2, len(eng._running)
    finally:
        eng.stop()
    for i, (s, o) in enumerate(zip(seqs, outs)):
        assert len(o) == 2, (i, o)
        assert o == [i + 2, i + 3], (i, o)
