"""统一规划：一个目标函数，一个决策者。

## 现状的问题

布局决策散在六处：resources.plan() 启动算一次、loading.try_grow() 首次 forward
后扩一次、autotune.PrefetchController 只调预取深度、engine 里若干处 shrink、
SLOT_FRACTION=0.90 硬编码、plan_quota() 从未被调用。

六套启发式互不知道对方存在，没有任何一处能回答「现在这个布局是不是最快的」。

## 目标函数

一步 decode 的时间：

    T = 权重搬运 + 专家装载 + 计算

      stream_s = stream_layers * layer_b / h2d_bps
      expert_s = uncovered * expert_b * (ram_frac/ram_bps + (1-ram_frac)/disk_bps)
      compute_s = 与常驻层数无关（每步都要算全部层）

**每一项都是资源量的函数。** 显存给常驻层 = 少搬权重；给专家槽位 = 少装专家；
给主机缓存 = 命中更多；给 pin = 走更快的搬运路径。三种用途争同一块显存，谁
更值钱由这个式子算，不需要 SLOT_FRACTION 这种比例常数。

这个式子里没有「240」「24」「40 层」这类模型特定数字 —— 它们都从 n_layers、
topk、layer_bytes、expert_bytes 和实测带宽推出来。

## 为什么这样就能覆盖「运行中调整」

式子里的三个带宽会随温度、别的进程、页缓存状态变。重测带宽、重解式子，就
得到新的最优布局。是否迁移取决于：迁移代价（= 要搬的字节 / 当前带宽）是不是
小于收益（新旧布局的 T 之差）。两者同量纲，可以直接比。

过热同时影响计算（降频）和带宽（PCIe 重训），两者都在 T 里，所以不需要为
「过热」单独写逻辑。
"""

from __future__ import annotations

from dataclasses import dataclass, field

GB = 1 << 30
MB = 1 << 20


@dataclass
class Layout:
    """一份布局。目标函数的变量就是这几个。"""

    resident: int = 0
    slots: int = 0
    cache_bytes: int = 0
    pin_ratio: float = 0.0
    embed_on_cpu: bool = False
    head_on_cpu: bool = False
    prefill_chunk: int = 8
    reason: str = ""


@dataclass
class Cost:
    """布局的代价分解（秒）。每项都能追到「多少字节 / 多少带宽」。"""

    stream_s: float = 0.0
    expert_s: float = 0.0
    compute_s: float = 0.0

    @property
    def total_s(self) -> float:
        return self.stream_s + self.expert_s + self.compute_s

    def parts(self) -> dict:
        return {"stream_s": round(self.stream_s, 4),
                "expert_s": round(self.expert_s, 4),
                "compute_s": round(self.compute_s, 4),
                "total_s": round(self.total_s, 4)}


class Planner:
    """统一规划与调整。不做 IO、不碰 CUDA —— 只接受资源量和实测带宽。"""

    def __init__(self, hw, model, bandwidth=None, calibrate=None):
        self.hw = hw
        self.model = dict(model or {})
        self.cal = dict(calibrate or {})
        if bandwidth is None:
            bandwidth = hw.bandwidth(self.cal) if hasattr(hw, "bandwidth") else {}
        self.h2d = float(bandwidth.get("h2d_bps") or 1.0)
        self.disk = float(bandwidth.get("disk_bps") or 1.0)
        self.disk_rand = float(bandwidth.get("disk_rand_bps") or self.disk)
        self.ram = float(bandwidth.get("ram_bps") or self.h2d)

        self.n_layers = int(self.model.get("n_layers") or 0)
        self.topk = int(self.model.get("topk") or 0)
        self.layer_b = float(self.model.get("layer_bytes") or 0.0)
        self.max_layer_b = float(self.model.get("max_layer_bytes") or self.layer_b)
        self.expert_b = float(self.model.get("expert_bytes") or 0.0)
        self.embed_b = float(self.model.get("embed_bytes") or 0.0)
        self.head_b = float(self.model.get("head_bytes") or 0.0)
        self.tied_head = bool(self.model.get("tied_head", False))

        caps = getattr(hw, "cards", None) or []
        self.vram = sum(c.total_gb for c in caps) * GB if caps else 0.0
        self.has_gpu = bool(caps) and self.vram > 0

    # ---- 目标函数 ---------------------------------------------------------

    def cost(self, lay: Layout) -> Cost:
        """估算这份布局的一步耗时。**唯一的判据。**"""
        n = self.n_layers
        stream = max(0, n - lay.resident)
        stream_s = stream * self.layer_b / max(1.0, self.h2d)

        # 一步要几个不同的 (layer, expert)；槽位覆盖掉多少，剩下的要搬。
        need = self.topk * n
        uncovered = max(0, need - lay.slots)
        if self.expert_b <= 0 or uncovered == 0:
            expert_s = 0.0
        else:
            # 命中主机缓存的走 ram→显存，未命中的走盘。
            total_experts = self.model.get("n_experts")
            if total_experts:
                all_b = floats(self.model, "expert_bytes") * n * int(total_experts)
            else:
                # 没有 n_experts 时按「缓存能装多少就命中多少」保守估：
                # 缓存的字节数 / 全部专家的字节数。全部按 n_layers*384 估。
                all_b = self.expert_b * n * 384
            hit = min(1.0, float(lay.cache_bytes) / max(1.0, all_b))
            per = self.expert_b
            expert_s = uncovered * per * (hit / max(1.0, self.ram)
                                          + (1.0 - hit) / max(1.0, self.disk_rand))

        # 计算：每步都要过全部层，与布局无关。用实测的每层计算耗时。
        compute_s = float(self.cal.get("compute_per_layer_s") or 0.0025) * n
        return Cost(stream_s=stream_s, expert_s=expert_s, compute_s=compute_s)

    # ---- 资源约束 ---------------------------------------------------------

    def used_bytes(self, lay: Layout) -> float:
        """这份布局在显存里占多少。"""
        b = 0.0
        b += lay.resident * self.layer_b * 1.7          # 权重 + activation/frag
        b += lay.slots * self.expert_b
        if not lay.embed_on_cpu:
            b += self.embed_b
        if not lay.head_on_cpu and not self.tied_head:
            b += self.head_b
        # 层搬运的周转缓冲：非 flat 路径每层一块（同一时刻一块），flat 路径
        # 是 BUF_COUNT 块。统一按「最坏一层 + 一个」算。
        if lay.resident < self.n_layers:
            b += self.max_layer_b * 2
        b += 0.20 * GB                                    # workspace
        return b

    def fits(self, lay: Layout) -> bool:
        return self.used_bytes(lay) <= self.vram * 0.94

    # ---- 求解 -------------------------------------------------------------

    def solve(self) -> Layout:
        """在资源约束下最小化 T。穷举 + 剪枝（维度只有 2，够用）。"""
        if not self.has_gpu:
            lay = Layout(resident=0, slots=0, embed_on_cpu=True, head_on_cpu=True)
            lay.reason = "no GPU: everything streams from host"
            return lay

        best = None
        # 常驻层从多到少试（多的先试，早剪枝）
        for resident in range(self.n_layers, -1, -1):
            # 给定 resident，槽位取「刚好装下的最大值」—— T 对 slots 单调递减
            for embed_cpu, head_cpu in ((False, False), (True, False), (True, True)):
                probe = Layout(resident=resident, slots=0,
                               embed_on_cpu=embed_cpu, head_on_cpu=head_cpu)
                free = self.vram * 0.94 - self.used_bytes(probe)
                if free < 0:
                    continue
                slots = int(free // max(1.0, self.expert_b)) if self.expert_b > 0 else 0
                lay = Layout(resident=resident, slots=slots,
                             embed_on_cpu=embed_cpu, head_on_cpu=head_cpu)
                if not self.fits(lay):
                    continue
                c = self.cost(lay)
                if best is None or c.total_s < best[0].total_s:
                    best = (c, lay)
            # 常驻层递减时 T 的权重搬运项在增加，槽位在减少 —— 不是单调的，
            # 所以不能提前 break。但一旦「搬权重的代价」已经超过理论最优的
            # 2 倍，后面的 resident 只会更差，可以停。
            if best is not None:
                lb = (self.n_layers - resident) * self.layer_b / max(1.0, self.h2d)
                if lb > best[0].total_s * 2:
                    break

        if best is None:
            lay = Layout(resident=0, slots=0, embed_on_cpu=True, head_on_cpu=True,
                         reason="nothing fits: streaming only")
            return lay
        c, lay = best
        lay.reason = ("T=%.3fs (stream %.3f + expert %.3f + compute %.3f)"
                      % (c.total_s, c.stream_s, c.expert_s, c.compute_s))
        return lay

    # ---- 运行中调整 -------------------------------------------------------

    def should_migrate(self, cur: Layout, new: Layout) -> tuple[bool, float]:
        """当前布局要不要迁到 new。返回 (迁移?, 净收益秒)。

        迁移代价 = 要搬的字节 / 当前带宽。和收益同量纲，直接比。
        """
        gain = self.cost(cur).total_s - self.cost(new).total_s
        moved = abs(new.resident - cur.resident) * self.layer_b
        cost = moved / max(1.0, self.h2d)
        # 只计算一次迁移代价，收益按「接下来 20 步」算 —— 步数少了不值得动。
        return (gain * 20 > cost, gain)


def floats(model, key):
    return float(model.get(key) or 0.0)
