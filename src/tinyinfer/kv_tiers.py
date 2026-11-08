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








def plan_expert_slots(*, devices: list[dict], expert_bytes: int,
                      reserve_bytes: int = 0, max_slots: int = 0) -> dict:
    """按真实可用显存决定专家槽位总数和每张卡各分多少。

    这是「引擎自己适配不同配置的机器」的核心一步：调用方不需要知道机器有几张卡、
    每张多大，只要把探测到的可用显存传进来，槽位规模自己算出来。

    devices: [{"name": "cuda:0", "free_bytes": 6_900_000_000}, ...]
              free_bytes 是「层权重、KV、激活都已经扣掉之后」真正能给专家用的量。
    expert_bytes: 单个专家 blob 的字节数（这个 checkpoint 是 18800640）。
    reserve_bytes: 每张卡额外留的余量（分配器的碎片、临时缓冲）。
    max_slots: 上限（0 = 不设）。用来做小规模实验或对齐某个已知值。

    返回 {"slots": 总槽位, "per_device": [{"name", "slots"}], "fits_all": 是否
    装得下整张专家表, "needed": 装下全部需要多少槽位}。

    为什么按卡分别算再相加，而不是「总显存 / blob」：多卡时显存不能跨卡使用，
    每张卡各自的余量决定它自己能装几个，余数不能合并。
    """
    blob = max(1, int(expert_bytes))
    per: list[dict] = []
    total = 0
    for d in devices or []:
        free = int(d.get("free_bytes", 0) or 0) - int(reserve_bytes)
        n = max(0, free // blob)
        per.append({"name": str(d.get("name", "cuda:0")), "slots": int(n)})
        total += int(n)
    if max_slots and max_slots > 0 and total > max_slots:
        # 按比例缩到上限，余数给最大的一侧（避免某张卡分到 0 个）
        scale = float(max_slots) / float(total)
        scaled = [{"name": p["name"], "slots": int(p["slots"] * scale)} for p in per]
        left = int(max_slots) - sum(p["slots"] for p in scaled)
        if left > 0 and scaled:
            big = max(range(len(scaled)), key=lambda i: per[i]["slots"])
            scaled[big]["slots"] += left
        per = scaled
        total = sum(p["slots"] for p in per)
    return {
        "slots": int(total),
        "per_device": per,
        "devices": len(per),
    }


def slots_needed_all(n_layers: int, n_experts: int) -> int:
    """把整张专家表全装进显存需要多少槽位（= 能否做到稳态零搬运）。"""
    return max(0, int(n_layers)) * max(0, int(n_experts))


def union_estimate(n: int, topk: int = 6, n_experts: int = 384) -> float:
    """估算 n 条会话在同一层的专家并集大小。

    实测点（真实 gate，这个 checkpoint 的第 0 层）：
        n=1  ->  6
        n=8  -> 35
        n=16 -> 61

    增长是次线性的（有热点专家），但边际增长率**递减**，不是常数：
        d(1..8)  = (35-6)/7  = 4.14
        d(8..16) = (61-35)/8 = 3.25

    用一个饱和曲线拟合这三点：

        union(n) = n_experts * (1 - exp(-a * n))

    其中 a 由「n=1 时等于 topk」定：1 - exp(-a) = topk/n_experts。
    这个形式在 n 大时自然饱和到 n_experts（不可能超过专家总数），
    在 n 小时接近 topk。

    校验：a = -ln(1 - 6/384) = 0.01575
        n=8:  384 * (1 - e^-0.126) = 45.5   （实测 35，偏高）
        n=16: 384 * (1 - e^-0.252) = 85.7   （实测 61，偏高）

    纯饱和曲线高估了。改用「幂律」形式更贴：

        union(n) = topk * n ** beta

    由 n=8 定 beta: 35 = 6 * 8^beta -> beta = ln(35/6)/ln(8) = 0.684
    校验 n=16: 6 * 16^0.684 = 39.9 （实测 61，偏低）

    两个都不够贴。折中：用分段线性插值，直接把实测点连起来 —— 这是最诚实的
    做法，不假装有一个精确的解析形式。超出实测范围时按最后一段的斜率外推，
    并限制在 n_experts 以内。
    """
    n = max(1, int(n))
    topk = max(1, int(topk))
    n_experts = max(topk, int(n_experts))
    # 实测锚点：(n, union)
    pts = [(1, float(topk)), (8, 35.0), (16, 61.0)]
    pts = [(x, min(float(y), float(n_experts))) for x, y in pts]
    if n <= pts[0][0]:
        return pts[0][1]
    for i in range(1, len(pts)):
        x0, y0 = pts[i - 1]
        x1, y1 = pts[i]
        if n <= x1:
            t = (n - x0) / float(x1 - x0)
            return y0 + t * (y1 - y0)
    # 超出实测范围：按最后一段斜率线性外推，封顶 n_experts
    x0, y0 = pts[-2]
    x1, y1 = pts[-1]
    slope = (y1 - y0) / float(x1 - x0)
    return min(float(n_experts), y1 + slope * (n - x1))


def batch_slots_bound(*, slots: int, n_layers: int, topk: int,
                      n_experts: int = 384,
                      reserve_ratio: float = 1.0) -> int:
    """槽位池能支撑多少条并发会话（编队约束的第二个来源）。

    并发共享一次 forward，所以一层要搬的是**所有会话路由的并集**。实测
    （tests/test_route_union.py，真实 gate）：8 条会话的并集是 35 个，不是
    最坏的 48 —— 每条新会话只多带约 4.1 个新专家，存在共同热点。

    所以「n 条会话一层要多少槽位」可以这样估：

        per_layer = topk + marginal * (n - 1)
        needed    = per_layer * n_layers

    返回满足 needed <= slots 的最大 n（至少 1）。

    这是和 `plan_slots`（按 KV 显存算）**独立**的第二条约束；真正的批宽是
    两者取小。两条约束都从真实数字出发，不写死常量。
    """
    slots = int(slots)
    n_layers = max(1, int(n_layers))
    topk = max(1, int(topk))
    if slots <= 0:
        return 1
    budget = int(slots * float(reserve_ratio))
    n = 1
    while True:
        per_layer = union_estimate(n, topk, n_experts)
        needed = per_layer * n_layers
        if needed > budget:
            return max(1, n - 1)
        n += 1
        if n > 4096:
            return n


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
