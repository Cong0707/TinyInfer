"""GPU 主动报缺、CPU 异步备货 —— 这一对是 decode 的关键路径。

角色分工（照用户的原话）：

  GPU 是买家：只负责说「我这一层缺哪几个专家」。判断在 GPU 上做（slot_ids
  本来就在显存里），结果写进一块 host-mapped 的环形队列 —— 主机和 GPU 映射
  同一块物理内存，GPU 写完 CPU 立刻能看见，中间没有 D2H 拷贝。

  CPU 是卖家：收到订单就备货。显存槽位里没有 → 看内存 → 内存也没有才读盘。
  备好之后在 compute stream 上写一个就绪计数（cuStreamWriteValue32），不占
  kernel 也不占主机时间。

  GPU 读权重之前等这个计数（cuStreamWaitValue32，等待发生在流里）。

这样一条 decode 路径上，主机侧**一次同步都没有**：CPU 布置完继续发下一层，
搬运和等待全部异步。原来的做法是每层把 miss 计数 item() 回主机（40 层 130ms）。
"""

from __future__ import annotations

import threading
from typing import Any, Iterable


class ExpertRequestRing:
    """GPU 写、CPU 读的请求环，映射在同一块物理内存上。

    布局（int32，全部 host-mapped）：
      [0]           发布计数（GPU atomicAdd）
      [1]           最新一条的序号（GPU 写，CPU 只读，用来判断有没有新东西）
      [2..]         每条请求 34 个 int：[layer, n, e0..e31, epoch]
    """

    SLOTS = 64
    STRIDE = 34
    HEAD = 2

    def __init__(self, device: str = "cuda"):
        import ctypes

        from .kernels import native

        self.device = device
        self.nbytes = (self.HEAD + self.SLOTS * self.STRIDE) * 4
        self._p, self.dev_ptr = native.host_mapped(self.nbytes)
        self._ctypes = ctypes
        U = ctypes.c_int32 * (self.nbytes // 4)
        self.host = U.from_address(self._p.value)
        for i in range(len(self.host)):
            self.host[i] = 0
        self._seen = 0
        self._lock = threading.Lock()

    # -- GPU 侧 ----------------------------------------------------------
    def post(self, slot_ids, n_exp: int, layer: int, epoch: int) -> None:
        """排一个 kernel 把本层缺的专家写进环（异步，不阻塞）。"""
        from .kernels import native

        fn = getattr(self, "_post_ln", None)
        if fn is None:
            fn = self._post_ln = native.BoundLaunch(
                native.kernel("moe_req", "moe_req_post"),
                (1,), (32,), "Tiiip")
        fn(slot_ids, int(n_exp), int(layer), int(epoch), int(self.dev_ptr))

    def posts_so_far(self) -> int:
        return int(self.host[0])

    # -- CPU 侧 ----------------------------------------------------------
    def drain(self) -> list[tuple[int, list[int]]]:
        """取走所有还没处理过的请求（不是阻塞调用，随时可以调）。"""
        out: list[tuple[int, list[int]]] = []
        with self._lock:
            head = int(self.host[1])
            if head <= self._seen:
                return out
            start = max(self._seen, head - self.SLOTS)
            for seq in range(start, head):
                rec = (self.HEAD + (seq % self.SLOTS) * self.STRIDE)
                layer = int(self.host[rec])
                n = int(self.host[rec + 1])
                if n <= 0 or n > 32:
                    continue
                ids = [int(self.host[rec + 2 + i]) for i in range(n)]
                out.append((layer, ids))
            self._seen = head
        return out

    def reset(self) -> None:
        with self._lock:
            for i in range(len(self.host)):
                self.host[i] = 0
            self._seen = 0

class ReadyFlag:
    """CPU 写、GPU 等的就绪计数（同样映射在同一块物理内存上）。"""

    def __init__(self, device: str = "cuda"):
        import ctypes

        from .kernels import native

        self.device = device
        self._p, self.dev_ptr = native.host_mapped(4)
        self.host = (ctypes.c_int32).from_address(self._p.value)
        self.host.value = 0
        self.value = 0

    def bump(self) -> int:
        """CPU 侧前进一格（备好一批就调一次）。"""
        self.value += 1
        self.host.value = self.value
        return self.value

    def wait_on_stream(self, target: int) -> None:
        """让 compute stream 等这一格（等待排在流里，主机不等）。

        优先用 cuStreamWaitValue32：不占 SM、纯硬件等待。Windows/WDDM 上
        这个 API 常常不支持（返回 801 CUDA_ERROR_NOT_SUPPORTED），那就退到
        自旋 kernel —— 语义一样，代价是一个 SM 被占住，而且只在数据确实还没
        到位时才真的转满。
        """
        from .kernels import native

        if getattr(self, "_memop_ok", None) is None:
            try:
                native.stream_wait_value32(self.dev_ptr, 0)   # 0 已满足，纯探测
                self._memop_ok = True
            except Exception:
                self._memop_ok = False
        if self._memop_ok:
            native.stream_wait_value32(self.dev_ptr, int(target))
            return
        fn = getattr(self, "_spin_ln", None)
        if fn is None:
            fn = self._spin_ln = native.BoundLaunch(
                native.kernel("moe_req", "moe_ready_spin"), (1,), (32,), "pi")
        fn(int(self.dev_ptr), int(target))

    def reset(self) -> None:
        self.value = 0
        self.host.value = 0


class ExpertServer:
    """GPU 是买家、CPU 是卖家：这个线程就是「卖家」。

    它循环做一件事：把 GPU 通过 host-mapped 环丢过来的缺失表取走，按
    显存 -> 内存 -> 磁盘 的顺序备货，每备好一批就把就绪计数加一。GPU 那边
    在流里等这个计数，主机侧一次都不用等。

    为什么是线程而不是在 forward 里同步做：forward 里同步做就是每层一次
    停顿（实测 130ms/token）。放到线程里之后，上一层 kernel 还在跑，这一层
    的专家已经在路上了 —— 这就是「CPU 不过多过问 GPU 干的活」。
    """

    def __init__(self, manager, ring: ExpertRequestRing, ready: ReadyFlag,
                 *, poll_us: int = 200):
        self.mgr = manager
        self.ring = ring
        self.ready = ready
        self.poll_us = int(poll_us)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._idle = threading.Event()
        self.served = 0
        self.batches = 0
        self.errors = 0

    def start(self) -> bool:
        if self._thread is not None:
            return False
        t = threading.Thread(target=self._run, name="expert-server", daemon=True)
        t.start()
        self._thread = t
        return True

    def stop(self) -> None:
        self._stop.set()
        t = self._thread
        if t is not None:
            t.join(timeout=5.0)
        self._thread = None

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                batch = self.ring.drain()
            except Exception:
                batch = []
            if not batch:
                self._idle.set()
                # 空闲时不要空转：CPU 该去把还没进内存的专家预热进来
                self._stop.wait(self.poll_us / 1e6)
                continue
            self._idle.clear()
            placed = 0
            for layer, ids in batch:
                try:
                    placed += int(self.mgr.need(layer, ids) or 0)
                    self.served += len(ids)
                except Exception:
                    self.errors += 1
            self.batches += 1
            # 备完了（不论是否真的搬了新的）就通知 GPU：它可以继续读这一批。
            try:
                self.ready.bump()
            except Exception:
                self.errors += 1

    def pending(self) -> int:
        try:
            return len(self.ring.drain())
        except Exception:
            return 0

    def stats(self) -> dict:
        return {"served": self.served, "batches": self.batches,
                "errors": self.errors, "ready": self.ready.value,
                "tiers": dict(getattr(self.mgr, "stats", {}))}

