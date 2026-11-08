"""GPU 报缺 -> CPU 备货 -> GPU 在流里等：MoE 每层只需要这三步。

这里把队列、卖家线程、就绪标志绑成一个对象，挂在 store 上（和 tier manager
一样，是「这块卡上的专家数据」的属性，不是某一层的属性）。
"""

from __future__ import annotations

import threading


class ExpertAsync:
    """一层一个入口，全机一个实例。

    典型用法（MoE 每层）：

        self._async.post(sids, n_exp, layer)   # 排一个 kernel，列出缺谁
        self._async.wait(layer)                # 让流等这批货（主机不等）

    然后 kernel 就可以照常读槽位表。第 k 层的等待目标是「第 k 次发布」，
    而发布是每层恰好一次、顺序确定的 —— 所以 CPU 侧完全不需要读 GPU 状态
    就知道该等多少。这就是「CPU 不过多过问 GPU 干的活」。
    """

    def __init__(self, manager, device: str = "cuda", *, enabled: bool = True):
        from .expert_queue import ExpertRequestRing, ExpertServer, ReadyFlag

        self.mgr = manager
        self.enabled = bool(enabled) and getattr(manager, "slots", None) is not None
        self.ring = ExpertRequestRing() if self.enabled else None
        self.ready = ReadyFlag() if self.enabled else None
        self.server = None
        self._base = 0        # 本实例累计发布过多少条
        self._posted = 0      # 本次 forward 里已经发布了几层
        self._lock = threading.Lock()
        if self.enabled:
            self.server = ExpertServer(manager, self.ring, self.ready, poll_us=200)

    def start(self) -> None:
        if self.server is not None:
            self.server.start()

    def stop(self) -> None:
        if self.server is not None:
            self.server.stop()

    # 一步（一个 token / 一个 prefill 批）开始，重置局部计数
    def step_begin(self) -> None:
        self._posted = 0

    def post(self, sids, n_exp: int, layer: int) -> None:
        """让 GPU 自己把「缺哪几个」写进环（kernel 级操作，不触发同步）。"""
        if not self.enabled:
            return
        self._posted += 1
        self.ring.post(sids, int(n_exp), int(layer), int(self._base + self._posted))

    def wait(self, layer: int | None = None) -> None:
        """在 compute stream 上等这一层发布的请求被备好。

        注意这不是「等主机」：排进流里就返回，主机继续往下发 kernel。
        真正的等待发生在 GPU 上（cuStreamWaitValue32，或退化成自旋 kernel）。
        """
        if not self.enabled:
            return
        target = int(self._base + self._posted)
        if target <= 0:
            return
        self.ready.wait_on_stream(target)

    def step_end(self) -> None:
        """一步结束：把本步发布过的条数记进基数。"""
        self._base += self._posted
        self._posted = 0

    def stats(self) -> dict:
        return {
            "enabled": self.enabled,
            "posted": self._base + self._posted,
            "ready": self.ready.value if self.ready is not None else 0,
            "server": self.server.stats() if self.server is not None else {},
        }
