"""CPU 空闲预热必须给推理让路。

预热线程读盘把专家填进内存。但如果推理同时在读同一块盘，两者抢带宽 ——
而磁盘正是低预算机器的瓶颈（L20 那台 3.0GB/s，一块 8GB 卡的 decode 全靠它）。
「没事干的时候加载」这句话的另一半含义是「有事干的时候就别加载」。
"""

import threading
import time

import pytest

from tinyinfer.expert_tiers import ExpertTierManager


class _Ram:
    capacity_entries = 8

    def __init__(self, delay=0.0):
        self.entries = {}
        self.delay = delay
        self.calls = 0

    def contains(self, layer, expert):
        return (layer, expert) in self.entries

    def get(self, layer, expert, blocking=True, admit=True):
        return self.entries.get((layer, expert))

    def admit(self, layer, expert, data):
        self.entries[(layer, expert)] = data
        return data

    def prefetch(self, layer, experts):
        self.calls += 1
        if self.delay:
            time.sleep(self.delay)
        for e in experts:
            self.entries[(int(layer), int(e))] = b"x"

        class _F:
            def result(self, timeout=None):
                return None

        return _F()


class _Reader:
    def read_many(self, blobs):
        return [b"x" for _ in blobs]


class _Site:
    def __init__(self):
        self.ram = _Ram(delay=0.02)
        self.slots = None
        self.index = object()
        # start_warm 要求 reader 存在，否则直接返回 False（没有读取器没法预热）
        self.reader = _Reader()
        self._tier_lock = threading.Lock()


def test_warm_waits_while_a_forward_is_running():
    st = _Site()
    # 数据要够多：预热跑完就没得让了。40 层 x 32 个专家 / 每 chunk 8 个 = 160 次
    mgr = ExpertTierManager(st, n_layers=40, n_experts=32, warm=False)
    assert mgr.start_warm(threads=1)
    try:
        mgr.layer_begin()          # 推理开始
        time.sleep(0.05)
        before = st.ram.calls
        time.sleep(0.15)
        assert st.ram.calls == before, (
            f"warm-up kept reading disk during a forward ({before} -> {st.ram.calls})")
        mgr.idle()                 # 推理结束
        deadline = time.time() + 2.0
        while st.ram.calls == before and time.time() < deadline:
            time.sleep(0.01)
        assert st.ram.calls > before, "warm-up never resumed after the forward"
    finally:
        mgr.stop_warm()
