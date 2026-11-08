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
        """按 内存 -> 磁盘 的顺序把字节找出来；能进内存的顺手进内存。"""
        out: dict[int, Any] = {}
        missing: list[int] = []
        for e in experts:
            data = None
            if self.ram is not None:
                try:
                    data = self.ram.get(layer, e, blocking=False)
                except Exception:
                    data = None
            if data is not None:
                out[e] = data
                self.stats["ram_hit"] += 1
            else:
                missing.append(e)
        if not missing:
            return out
        # 内存里没有：CPU 去盘上取（批量 pread，一个专家一次读）。
        try:
            blobs = [self.index.expert_blob(layer, e) for e in missing]
            datas = self.reader.read_many(blobs)
        except Exception:
            return out
        for e, d in zip(missing, datas):
            if d is None:
                continue
            out[e] = d
            self.stats["disk"] += 1
            if self.ram is not None:
                try:
                    self.ram.admit(layer, e, d)
                except Exception:
                    pass
        return out

    def need(self, layer: int, experts: Iterable[int], *, urgent: bool = True) -> int:
        """备货（可以在任何线程）：把缺失的专家字节取出来、挑好槽位。

        只做主机侧的事：查内存、读盘、挑槽位、把字节拷进 pinned 中转区。
        **不发任何 CUDA 调用** —— 卖家线程跨线程发 CUDA 会和主线程互等死锁。

        urgent=False 是预取：**只在有空槽位时**才占位，绝不为它覆盖别人正在用的
        槽位。这一条是必须的 —— 预取和当前层抢槽位时，输的必须是预取：实测
        不加区分地预取会让当前层自己都装不进去（"still not resident"）。
        """
        if self.slots is None or not self.slots.slots:
            return 0
        want = [int(e) for e in experts]
        miss = [e for e in want if (layer, e) not in self.slots._by_key]
        self.stats["vram_hit"] += len(want) - len(miss)
        if not miss:
            return 0
        self.stats["miss"] += len(miss)
        data = self._provide(layer, miss)
        if not data:
            return 0
        staged = self.slots.stage(
            layer, list(data.keys()),
            lambda lay, want_e: {e: data[e] for e in want_e if e in data},
            urgent=urgent)
        self.stats["placed"] += int(staged)
        return int(staged)

    def commit(self, layer: int | None = None) -> int:
        """把备好的货发出去（主线程调，只在本线程 stream 上）。"""
        if self.slots is None:
            return 0
        return int(self.slots.commit(layer))


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
        if self.slots is not None:
            self.slots.clear_urgent()
        # 告诉预热线程「我在忙」：它会在下一个 chunk 之前让路。
        # 注意不在每层结束时清掉 —— 层间没有真正的空档，一轮 40 层连着跑，
        # 清了就等于没让路。真正的空闲是「两次 forward 之间」，由 engine 清。
        self._busy.set()

    def idle(self) -> None:
        """整轮 forward 结束：预热线程可以去做「没事干时该做的事」了。"""
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

    def ready(self, layer: int, experts: Iterable[int]) -> None:
        """让 compute stream 等这批专家的 DMA。已落地的槽位不插等待。"""
        if self.slots is None:
            return
        idxs = []
        for e in experts:
            idx = self.slots._by_key.get((layer, int(e)))
            if idx is not None:
                idxs.append(idx)
        if idxs:
            self.slots.wait_ready(idxs)

    def release(self, layer: int, experts: Iterable[int]) -> None:
        """本层算完归还引用（槽位与数据仍留着，下一步命中免搬）。"""
        if self.slots is None:
            return
        for e in experts:
            try:
                self.slots.release(layer, int(e))
            except Exception:
                pass

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
            for layer in range(self.n_layers):
                if stop.is_set():
                    return
                gap = [e for e in range(self.n_experts) if not self.ram.contains(layer, e)]
                if not gap:
                    continue
                n = max(1, threads)
                step = max(1, len(gap) // n)
                mine = gap[worker * step:] if worker == n - 1 else gap[worker * step:(worker + 1) * step]
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
