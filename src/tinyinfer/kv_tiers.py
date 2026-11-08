"""KV 分层的容量决策：1M 上下文一条请求就是约 7GB，不能按上限预分配。

实测（本机按模型真实结构算）：
  N=128k  compressed KV = 0.89 GB
  N=1M    compressed KV = 6.92 GB
而 L20 只有 44GB，扣掉 40 层权重（6.5GB）、激活、专家槽位后，剩给 KV 的
大概 30GB —— 按上限预分配的话，一条 1M 请求就把并发打成 1。

所以 KV 也必须走同一套「按容量覆盖」：
  - 显存里放的是**当前活跃**的那部分；
  - 不活跃的历史沉到内存（371GB，放得下几十条 1M 会话）；
  - 用到哪一段才把哪一段拉回来。

注意这和专家槽位不同：KV 是**有序的**，不能按 LRU 乱覆盖 —— 覆盖掉中间一段
就是上下文断裂。所以 KV 的分层单位是「整段连续区间」，不是单格。
"""

from __future__ import annotations

import threading


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

    def describe(self) -> dict:
        return {"vram_free_gb": self.vram_free / 2**30,
                "reserve_gb": self.reserve / 2**30,
                "kv_budget_gb": self.kv_budget() / 2**30}
