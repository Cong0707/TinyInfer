"""显存 / 内存 / 磁盘 的唯一决策点。

以前这条链路散在三处：`VramSlotCache`（槽位）、`ExpertCache`（内存）、
`ShardReader`（磁盘），而「该搬谁、从哪搬、搬完表写哪」写在 `deepseek/moe.py`
里。每加一个功能就要在 forward 里补一遍回收/清表，修一处崩一处 —— 这不是
注释能解决的问题，是职责放错了地方。

这里把它收成一个对象。调用方（MoE）只做两件事：

    mgr.need(layer, experts)     # 我这一层要这几个人
    mgr.ready(layer, experts)    # 用之前等它们的 DMA 落地

其余（显存有没有、内存有没有、要不要读盘、读到的字节进不进内存、槽位怎么
选、表怎么写、什么时候算搬完）都在这里。

分级顺序就是用户说的买卖关系：GPU 是买家（只报缺谁），CPU 是卖家（自家仓库
= 内存，上游供货 = 磁盘）。买家不过问卖家怎么备货。
"""

from __future__ import annotations

import threading
import time
from typing import Any, Iterable


class ExpertTierManager:
    """一块卡上专家数据的完整生命周期。"""

    def __init__(self, store, *, n_layers: int = 40, n_experts: int = 384,
                 device: str = "cuda", warm: bool = True):
        self.store = store
        self.slots = getattr(store, "slots", None)
        self.ram = getattr(store, "ram", None)
        self.index = getattr(store, "index", None)
        self.reader = getattr(store, "reader", None)
        self.n_layers = int(n_layers)
        self.n_experts = int(n_experts)
        self.device = device
        self._lock = threading.Lock()
        self._source_pending: dict[tuple[int, int], int] = {}
        self._source_flights: list[tuple[tuple[int, int], Any]] = []
        # 每一级命中一次就记一次：这几个计数是「引擎到底在干什么」的唯一证据，
        # 出问题时不看代码看数字。
        self.stats = {"vram_hit": 0, "ram_hit": 0, "disk": 0,
                      "placed": 0, "miss": 0, "warm": 0}
        self._warm_stop = threading.Event()
        self._warm_thread: Any = None
        # 推理正在跑的时候不要读盘：预热是「没事干才做的事」，和 decode 抢
        # 磁盘只会把关键路径拖慢（实测磁盘是低预算机器的瓶颈）。
        self._busy = threading.Event()
        if warm:
            self.start_warm()

    # -- 查询 -------------------------------------------------------------
    def resident(self, layer: int, expert: int) -> bool:
        if self.slots is None:
            return False
        return (layer, expert) in self.slots._by_key

    def table(self):
        """device 侧的 [n_layers, n_experts] int32 槽位表，-1 表示不在显存。"""
        if self.slots is None:
            return None
        return self.slots.slot_table(self.n_layers, self.n_experts)

    # -- 备货 -------------------------------------------------------------
    def _provide(self, layer: int, experts: list[int]) -> dict[int, Any]:
        """明确区分缓存命中与磁盘读取，任何已选路径失败都直接抛出。"""
        out = {}
        missing = []
        for expert in experts:
            data = self.ram.get(layer, expert, blocking=False) if self.ram is not None else None
            if data is None:
                missing.append(expert)
            else:
                out[expert] = data
                self.stats["ram_hit"] += 1
        if missing:
            blobs = [self.index.expert_blob(layer, expert) for expert in missing]
            pooled = getattr(self.ram, "_read_many_pooled", None)
            data = pooled(blobs) if pooled is not None else self.reader.read_many(blobs)
            if len(data) != len(missing) or any(value is None for value in data):
                raise RuntimeError(f"layer {layer}: expert source read returned incomplete data")
            for expert, value in zip(missing, data):
                out[expert] = value
                self.stats["disk"] += 1
                if self.ram is not None:
                    self.ram.admit(layer, expert, value)
        return out

    def need(self, layer: int, experts: Iterable[int], *, urgent: bool = True) -> int:
        """供货整个实际请求，同时持有缓存源直到 H2D 完成。"""
        want = list(dict.fromkeys(int(expert) for expert in experts))
        if not want:
            return 0
        if self.slots is None or not self.slots.slots:
            raise RuntimeError(f"layer {layer}: expert supply requires a planned slot pool")
        with self._lock:
            self._drain_sources_locked(wait=False)
            with self.slots._lock:
                resident = sum((layer, expert) in self.slots._by_key for expert in want)
            self.stats["vram_hit"] += resident
            self.stats["miss"] += len(want) - resident
            held = []

            def provide(lay, missing):
                if self.ram is not None:
                    for expert in missing:
                        key = (lay, expert)
                        self.ram.lease(key, 1)
                        held.append(key)
                return self._provide(lay, missing)

            try:
                staged = self.slots.stage(layer, want, provide, urgent=urgent)
            except BaseException:
                for key in held:
                    self.ram.lease(key, -1)
                raise
            with self.slots._lock:
                for key in held:
                    index = self.slots._by_key.get(key)
                    if index is not None and index in self.slots._stage_bufs:
                        self._source_pending[key] = index
                    else:
                        self.ram.lease(key, -1)
            self.stats["placed"] += staged
            return staged

    def commit(self, layer: int | None = None) -> int:
        """提交 DMA 后按实际完成事件保留源缓存租约，包括异常退出。"""
        if self.slots is None:
            raise RuntimeError("expert commit requires a planned slot pool")
        with self._lock:
            try:
                return self.slots.commit(layer)
            finally:
                for key, index in list(self._source_pending.items()):
                    if layer is None or key[0] == layer:
                        self._source_flights.append((key, self.slots._copy_events[index]))
                        del self._source_pending[key]
                self._drain_sources_locked(wait=False)

    def _drain_sources_locked(self, *, wait: bool) -> None:
        index = 0
        while index < len(self._source_flights):
            key, event = self._source_flights[index]
            if event is not None and wait:
                event.synchronize()
            if event is None or wait or event.query():
                self.ram.lease(key, -1)
                self._source_flights.pop(index)
            else:
                index += 1

    def drain_sources(self, *, wait: bool = False) -> None:
        with self._lock:
            self._drain_sources_locked(wait=wait)

    def close(self) -> None:
        self.stop_warm()
        with self._lock:
            for key in list(self._source_pending):
                self.slots.cancel_request(key[0], [key[1]])
                self.ram.lease(key, -1)
                del self._source_pending[key]
            self._drain_sources_locked(wait=True)

    # -- 给 MoE 的薄接口：它不该知道池子的内部结构 -----------------------
    @property
    def blob_bytes(self) -> int:
        return int(getattr(self.slots, "blob_bytes", 0))

    def table(self):
        """device 侧的 [n_layers, n_experts] 槽位表（-1 = 不在显存）。"""
        if self.slots is None:
            return None
        return self.slots.slot_table(self.n_layers, self.n_experts)

    def layer_begin(self) -> None:
        """一层开始：它要用的专家从这一刻起不许被别的层顶掉。"""
        # 请求的保护在 ready/release 或取消事务时解除，不随层切换清空。
        # 告诉预热线程「我在忙」：它会在下一个 chunk 之前让路。
        # 注意不在每层结束时清掉 —— 层间没有真正的空档，一轮 40 层连着跑，
        # 清了就等于没让路。真正的空闲是「两次 forward 之间」，由 engine 清。
        self._busy.set()

    def idle(self) -> None:
        """整轮 forward 结束：预热线程可以去做「没事干时该做的事」了。"""
        self.drain_sources()
        self._busy.clear()

    def capabilities(self) -> dict:
        """这一套布局能做什么，供上层（EP / 规划器）自适应。

        不写死任何数字：槽位数、内存容量、是否多卡，全问实际对象。
        """
        devices = list(getattr(self.slots, "devices", None) or []) if self.slots else []
        _slots = int(getattr(self.slots, "slots", 0) or 0)
        # 三个不同量级的「需要多少槽位」，别混用：
        #   slots_needed_all        全部专家（40x384=15360），理论上限，没人能达到
        #   slots_needed_a_step     一步 decode 要用的专家（n_layers x topk = 240）
        #                           —— **这才是「稳态零搬运」的真实门槛**
        _need_all = self.n_layers * self.n_experts
        _topk = max(1, int(getattr(self, "topk", 6) or 6))
        _need_step = self.n_layers * _topk
        _one_layer = self.n_experts
        return {
            "slots": _slots,
            "blob_bytes": self.blob_bytes,
            "vram_gb": _slots * self.blob_bytes / 2**30,
            "devices": devices,
            "n_devices": max(1, len(devices)),
            "ram_entries": int(getattr(self.ram, "capacity_entries", 0) or 0)
                            if self.ram is not None else 0,
            # ---- 这层配置「够不够」的判断依据，给上层自适应用 ----
            # 全驻留需要的槽位：一层如果用 topk 个专家，n_layers*topk 就是下限
            "slots_needed_all": _need_all,
            "slots_needed_a_step": _need_step,
            "slots_needed_one_layer": _one_layer,
            # 能不能让**一步 decode** 需要的 n_layers*topk 个专家同时驻留。
            # False 意味着 decode 每步都要搬（8GB 卡的 14 槽位 < 240）。
            "can_hold_a_step": bool(_slots >= _need_step),
            # 能不能让**一层**的专家全驻留（更弱的要求）
            "can_hold_a_layer": bool(_slots >= _one_layer),
        }

    def ready(self, layer: int, experts: Iterable[int]) -> list[int]:
        """完整请求一次取得读租约，再让计算流等待各自的 DMA。"""
        wanted = list(dict.fromkeys(int(expert) for expert in experts))
        if not wanted:
            return []
        if self.slots is None:
            raise RuntimeError(f"layer {layer}: requested experts have no slot pool")
        indices = self.slots.acquire_ready(layer, wanted)
        try:
            self.slots.wait_ready(indices)
        except BaseException:
            self.slots.release_many(layer, wanted)
            raise
        return wanted

    def release(self, layer: int, experts: Iterable[int]) -> None:
        """本层算完在消费者流记完成点，再归还读租约。"""
        if self.slots is None:
            raise RuntimeError(f"layer {layer}: releasing experts without a slot pool")
        wanted = list(dict.fromkeys(int(expert) for expert in experts))
        if wanted:
            self.slots.release_many(layer, wanted)

    # -- 空闲预热 ---------------------------------------------------------
    def start_warm(self, threads: int = 2) -> bool:
        """CPU 没事干的时候把还没进内存的专家读进来。

        显存放不下整个专家集，内存放得下（371GB 对 289GB）。预热完之后
        每次搬专家只剩 RAM->VRAM 的 DMA，磁盘彻底退出关键路径 —— 这是低
        预算机器能撑并发的前提。
        """
        if self._warm_thread is not None or self.ram is None or self.reader is None:
            return False
        stop = self._warm_stop

        def _one(worker: int) -> None:
            # **按「专家位置」跨层轮转，而不是逐层填满。**
            #
            # 以前是 `for layer in range(40)` 顺序填：填到第 6 层时，第 30 层的
            # decode 还是空的，于是每层都去读盘。实测 L20：预热 76 个专家/秒、
            # 装完 13150 个要 173 秒，而头 60 秒里 decode 一直读盘 ——
            # GPU 利用率 0-16%，CPU 189%，每个 step 500-3000ms。
            #
            # decode 的访问模式是「每层都要几个」，所以正确的顺序是
            # (第 0 层专家0..7) -> (第1层专家0..7) -> ... -> (第39层专家0..7)
            # -> (每层专家 8..15) ...
            # 这样预热到 1/N 时，每层都有 1/N 的覆盖，命中率随进度线性上升，
            # 而不是「前几层 100%、其余 0%」。
            #
            # 每层的专家按 8 个一组切，外层遍历组号、内层遍历层。
            _GRP = 8
            _groups = (self.n_experts + _GRP - 1) // _GRP
            for g in range(_groups):
                if stop.is_set():
                    return
                for layer in range(self.n_layers):
                    if stop.is_set():
                        return
                    lo = g * _GRP
                    hi = min(self.n_experts, lo + _GRP)
                    mine = [e for e in range(lo, hi) if not self.ram.contains(layer, e)]
                    if not mine:
                        continue
                    for i in range(0, len(mine), 8):
                        if stop.is_set():
                            return
                        # 推理在忙的时候让路：预热是「没事干才做的事」，和 decode 抢
                        # 磁盘只会把关键路径拖慢。_busy 由 MoE 每层置位。
                        # 注意不能用 _busy.wait()：那是等「被置位」，而这里要等的是
                        # 「被清空」。事件已置位时 wait() 立刻返回，等于没让路。
                        while self._busy.is_set():
                            if stop.is_set():
                                return
                            time.sleep(0.005)
                        chunk = mine[i:i + 8]
                        try:
                            self.ram.prefetch(layer, chunk).result(timeout=300)
                            self.stats["warm"] += len(chunk)
                        except Exception:
                            return

        def _run() -> None:
            ts = [threading.Thread(target=_one, args=(w,), name=f"expert-warm{w}",
                                   daemon=True) for w in range(max(1, threads))]
            for t in ts:
                t.start()
            for t in ts:
                t.join()

        t = threading.Thread(target=_run, name="expert-warm", daemon=True)
        self._warm_thread = t
        t.start()
        return True

    def stop_warm(self) -> None:
        self._warm_stop.set()
        thread = self._warm_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join()

    def warm_progress(self) -> float:
        if self.ram is None:
            return 1.0
        try:
            total = (self.ram.capacity_entries if hasattr(self.ram, "capacity_entries")
                     else self.n_layers * self.n_experts)
            have = len(getattr(self.ram, "_entries", {}))
            return min(1.0, have / max(1, total))
        except Exception:
            return 0.0
