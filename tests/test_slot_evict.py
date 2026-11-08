"""会话换出时的 KV 交接：SlotManager 把会话挪走，服务端负责把 KV 沉到内存。

以前 SlotManager 挤掉一个会话只是从 _sessions 里删掉，KV 直接丢 —— 那个会话
下次回来要整段重新 prefill。1M 上下文意味着几分钟的重复计算，这是「并发」
和「长上下文」直接冲突的地方。
"""

import pytest

from tinyinfer.server import SlotManager


def _mgr(max_slots=2):
    return SlotManager(max_slots=max_slots, max_context_tokens=1024)


def test_evict_callback_fires_when_a_slot_is_recycled():
    m = _mgr(2)
    seen = []
    m.on_evict = lambda sid, slot: seen.append((sid, slot))
    s0, _ = m.acquire("a")
    m.release(s0)
    s1, _ = m.acquire("b")
    m.release(s1)
    # 第三个会话必须挤掉一个，换出回调应该被调用
    s2, _ = m.acquire("c")
    m.release(s2)
    assert seen, "evicting a session must hand its KV back to the caller"
    assert seen[0][0] in ("a", "b"), seen


def test_no_evict_callback_keeps_old_behaviour():
    m = _mgr(2)
    s0, _ = m.acquire("a")
    m.release(s0)
    s1, _ = m.acquire("b")
    m.release(s1)
    s2, _ = m.acquire("c")      # 没接回调也不能炸
    m.release(s2)
    assert m.snapshot()["sessions"] >= 1


def test_evict_callback_exception_does_not_block_the_request():
    m = _mgr(2)

    def boom(sid, slot):
        raise RuntimeError("swap failed")

    m.on_evict = boom
    s0, _ = m.acquire("a")
    m.release(s0)
    s1, _ = m.acquire("b")
    m.release(s1)
    s2, _ = m.acquire("c")      # 换出失败也要让新请求继续
    assert s2 is not None
