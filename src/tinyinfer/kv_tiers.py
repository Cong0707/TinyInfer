"""会话 KV 的分层换入换出：卡上只放活跃的，其余沉到内存。

为什么需要：实测一条 1M 上下文的压缩 KV 是 6.91GB。L20 的 44GB 扣掉 40 层
权重、激活、专家槽位后剩约 30GB —— 不做换出的话，卡上同时能驻留的会话数
就是 4。而真实服务要的是「很多个长会话轮流推进」，不是「4 个跑完再换」。

内存有 371GB，放得下五十多条 1M 会话的 KV。所以：
  - 卡上是「当前这一步真正在算的」那几条；
  - 其余的整条换出到内存（D2H 一次、H2D 一次，按需）；
  - 换出/换入是整段连续区间，不做 LRU 单格覆盖 —— KV 是有序的，
    覆盖掉中间一段就是上下文断裂。

这一层和专家槽位用的是两套策略，但**同一个容量来源**：都问「这块卡现在
还剩多少」，而不是写死数字。
"""

from __future__ import annotations

import threading
from typing import Any




class KvTierPlan:
    """决定一条会话的 KV 能占多少显存。

    不写死字节数：问的是「这块卡上还剩多少」，和专家槽位、层权重的分配用
    同一套判断。换机器不需要改任何参数。
    """

    def __init__(self, *, vram_free_bytes: int = 0, reserve_bytes: int = 0):
        self.vram_free = max(0, int(vram_free_bytes))
        self.reserve = max(0, int(reserve_bytes))

    def kv_budget(self) -> int:
        return max(0, self.vram_free - self.reserve)

    def groups_that_fit(self, head_dim: int, bytes_per_value: int = 1,
                        share: float = 0.6) -> int:
        """按每个压缩组 head_dim 字节算，这块预算能放多少组。

        share 是「KV 占可用余量的比例」：还要给激活、临时张量留地方。
        """
        per = max(1, int(head_dim) * max(1, int(bytes_per_value)))
        return int(self.kv_budget() * float(share)) // per

    def sessions_that_fit(self, bytes_per_session: int) -> int:
        """这块预算能同时驻留几条会话（并发能力的上界）。"""
        n = max(1, int(bytes_per_session))
        return self.kv_budget() // n

    def describe(self) -> dict:
        return {"vram_free_gb": self.vram_free / 2**30,
                "reserve_gb": self.reserve / 2**30,
                "kv_budget_gb": self.kv_budget() / 2**30}




def plan_slots(*, vram_free_bytes: int = 0, head_dim: int = 512,
                n_layers: int = 40, context_tokens: int = 1_048_576,
                ratio_mix=(0, 5), ratio_a=(2, 18), ratio_b=(1, 20),
                reserve_bytes: int = 2 * 2**30, min_slots: int = 1) -> int:
    """根据**这块卡真实剩多少显存**决定并发槽位数，而不是写死 4。

    一条会话的 KV 有多大是可以算的（压缩 latent，fp4 存储）：
    每层每个压缩组占 head_dim/2 字节，1M 上下文下 --
    实测本 checkpoint 是 6.91GB，所以 44GB 卡扣掉权重/激活后大约是 4-6 条。

    同一份代码在 8GB 卡上算出来就是 1 条，在 141GB 卡上是几十条 ——
    **规模由硬件决定，不是人工配参数**。
    """
    if context_tokens <= 0:
        return max(1, int(min_slots))
    per_layer_bytes = 0
    for ratio, count in (ratio_mix, ratio_a, ratio_b):
        if ratio > 0:
            per_layer_bytes += int(count) * (context_tokens // int(ratio)) * (head_dim // 2)
    if per_layer_bytes <= 0:
        return max(1, int(min_slots))
    budget = max(0, int(vram_free_bytes) - int(reserve_bytes))
    n = int(budget // max(1, per_layer_bytes))
    return max(int(min_slots), n)


class KvSwapPool:
    """按「整条会话」换出 KV 的主机侧仓库。

    entries 里存的是已经搬到 CPU 的张量（pinned 更好：回搬时能走异步 DMA）。
    容量按字节记账，超了就按最久没用的换出去 —— 注意这是**整条会话**的粒度，
    不是它内部的某一段，所以顺序性不会被破坏。
    """

    def __init__(self, capacity_bytes: int = 0):
        self.capacity = max(0, int(capacity_bytes))
        self._entries: dict[str, tuple[dict, int]] = {}   # key -> (snap, nbytes)
        self._order: list[str] = []
        self._used = 0
        self._lock = threading.Lock()
        self.swapped_out = 0
        self.swapped_in = 0
        self.dropped = 0

    @staticmethod
    def _nbytes(snap: Any) -> int:
        total = 0
        try:
            items = snap.values() if hasattr(snap, "values") else []
            for v in items:
                if hasattr(v, "numel"):
                    total += int(v.numel() * v.element_size())
                elif isinstance(v, (tuple, list)):
                    for t in v:
                        if hasattr(t, "numel"):
                            total += int(t.numel() * t.element_size())
        except Exception:
            pass
        return total

    def put(self, key: str, snap: dict) -> int:
        """把一条会话的 KV 放进主机仓库（调用方已经搬到 CPU）。"""
        n = self._nbytes(snap)
        with self._lock:
            old = self._entries.get(key)
            if old is not None:
                self._used -= old[1]
                try:
                    self._order.remove(key)
                except ValueError:
                    pass
            self._entries[key] = (snap, n)
            self._used += n
            self._order.append(key)
            self.swapped_out += 1
            # 满了就丢最久没用的整条会话（调用方负责在丢之前落盘）
            while self.capacity and self._used > self.capacity and len(self._order) > 1:
                victim = self._order.pop(0)
                ent = self._entries.pop(victim, None)
                if ent is not None:
                    self._used -= ent[1]
                    self.dropped += 1
            return n

    def get(self, key: str) -> dict | None:
        with self._lock:
            ent = self._entries.get(key)
            if ent is None:
                return None
            try:
                self._order.remove(key)
            except ValueError:
                pass
            self._order.append(key)
            self.swapped_in += 1
            return ent[0]

    def drop(self, key: str) -> bool:
        with self._lock:
            ent = self._entries.pop(key, None)
            if ent is None:
                return False
            self._used -= ent[1]
            try:
                self._order.remove(key)
            except ValueError:
                pass
            return True

    def stats(self) -> dict:
        return {"entries": len(self._entries),
                "used_gb": self._used / 2**30,
                "capacity_gb": self.capacity / 2**30,
                "swapped_out": self.swapped_out,
                "swapped_in": self.swapped_in,
                "dropped": self.dropped}


def to_host(snap: dict) -> dict:
    """把一份 KV snapshot 搬到 CPU（非阻塞留着给下一次 forward 用）。"""
    out = {}
    for k, v in snap.items():
        if hasattr(v, "to"):
            out[k] = v.to("cpu", non_blocking=False)
        elif isinstance(v, tuple):
            out[k] = tuple(t.to("cpu", non_blocking=False) if hasattr(t, "to") else t
                           for t in v)
        else:
            out[k] = v
    return out
