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
