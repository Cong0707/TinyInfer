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

# tilelang 的 workspace：固定部分 + 每个 token 的增量。
# 引擎要用它来给槽位池留运行期余量（拿满 free 会在 activation 上 OOM）。
# 这个量是「运行时的真实需求」，不是机器常数 —— 它由 kernel 的 tile 形状定，
# 换模型（行长不同）会变，所以放在规划器里作为可标定项。
WS_BASE = 0.20 * GB
WS_PER_TOKEN = 0.02 * GB


@dataclass
class Layout:
    """一份布局。目标函数的变量就是这几个。"""

    # resident 是权威字段，engine 通过 gpu_layers 属性读它（同名不同叫法）。
    resident: int = 0
    slots: int = 0
    cache_bytes: int = 0
    pin_ratio: float = 0.0
    embed_on_cpu: bool = False
    head_on_cpu: bool = False
    prefill_chunk: int = 8
    reason: str = ""
    # 引擎还会读这几个。名字沿用旧 Plan 的，因为它们描述的是同一件事 ——
    # 不是为了兼容，而是 engine 里 15 处引用描述的就是这些量。
    stream_layers: int = 0
    read_threads: int = 16
    fits: bool = True
    speculation: bool = False

    @property
    def gpu_layers(self) -> int:
        """旧 Plan 的名字。engine 用它读常驻层数。"""
        return int(self.resident)

    @property
    def expert_slots(self) -> int:
        """旧 Plan 的名字。engine 用它读专家槽位数。"""
        return int(self.slots)


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

    def __init__(self, hw, model, bandwidth=None, calibrate=None,
                 model_dir=None):
        self.hw = hw
        self.model = dict(model or {})
        self.model_dir = str(model_dir or "")
        self.cal = dict(calibrate or {})
        # **带宽优先级：调用方显式给 > 训练器实测存下的 > 现场测**
        #
        # 训练器（tinyinfer.tune）测多次取中位数后写进 tuning.json；那是比
        # 单次现场测量更可信的值（单次可能撞页缓存，disk 能差一倍）。
        # 所以这里是「读配置」而不是每次都重测 —— 重测留给训练器。
        if bandwidth is None:
            stored = {}
            try:
                import tinyinfer.tuning as _tn

                _p = _tn.params(getattr(self, "model_dir", "")) or {}
                stored = {k: _p[k] for k in
                          ("h2d_bps", "disk_bps", "disk_rand_bps", "ram_bps")
                          if _p.get(k)}
            except Exception:
                stored = {}
            if stored and stored.get("h2d_bps"):
                bandwidth = stored
            elif hasattr(hw, "bandwidth"):
                bandwidth = hw.bandwidth(self.cal)
            else:
                bandwidth = {}
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

        # ---- 训练器可调的三个量 ------------------------------------------
        #
        # 这三个**不由「资源量 ÷ 单元大小」推出**，必须实测，所以它们是参数
        # 而不是求解结果。训练器（tinyinfer.tune）扫它们，写进 tuning.json，
        # 这里优先读环境变量（训练器扫参用），其次读 tuning，最后用初值。
        #
        # 注意：resident / slots / cache_bytes **不在**这里 —— 它们由目标函数
        # 解出来，不是可调参数。以前把 slots 做成可调导致规划器被上一轮的旧值
        # 永久锁死。
        import os as _os
        self.compute_per_layer_s = self._tunable(
            "compute_per_layer_s", 0.0025, float)
        self.pin_ratio = self._tunable("pin_ratio", 0.5, float)
        self.ws_base_gb = self._tunable("ws_base_gb", None, float)

    def _tunable(self, name, default, cast):
        """读一个可调量：环境变量（训练器用）> calibrate > 默认。"""
        import os as _os

        env_key = "TI_TUNE_" + name.upper()
        v = _os.environ.get(env_key)
        if v not in (None, ""):
            try:
                return cast(v)
            except (TypeError, ValueError):
                pass
        v = self.cal.get(name)
        if v is not None:
            try:
                return cast(v)
            except (TypeError, ValueError):
                pass
        return default

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
                all_b = self.expert_b * n * int(total_experts)
            else:
                # 没有 n_experts 时按「缓存能装多少就命中多少」保守估：
                # 缓存的字节数 / 全部专家的字节数。全部按 n_layers*384 估。
                all_b = self.expert_b * n * 384
            hit = min(1.0, float(lay.cache_bytes) / max(1.0, all_b))
            per = self.expert_b
            expert_s = uncovered * per * (hit / max(1.0, self.ram)
                                          + (1.0 - hit) / max(1.0, self.disk_rand))

        # 计算：每步都要过全部层，与布局无关。用实测的每层计算耗时。
        compute_s = float(self.compute_per_layer_s or 0.0025) * n
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
            # 剪枝：resident 继续减，stream_s 只会单调增（搬更多层），而 expert_s
            # 最多降到 0（槽位覆盖全部）。所以「已经确定的下界」是
            #
            #   lb = 未来最少的搬运 + 0（专家全命中）+ compute
            #
            # 一旦 lb >= 当前最优，后面的 resident 不可能更好，可以停。
            #
            # 原来的剪枝写的是 `lb > best * 2`，两倍太松 —— 实测会漏掉更优解
            # （4层/246槽位 被选为最优，而 8层/240槽位 更便宜）。
            if best is not None:
                lb = (self.n_layers - resident) * self.layer_b / max(1.0, self.h2d)
                lb += float(self.cal.get("compute_per_layer_s") or 0.0025) * self.n_layers
                if lb >= best[0].total_s:
                    break

        if best is None:
            # 连「全部流式 + 零槽位」都装不下：这张卡放不下这个模型的骨架。
            # 如实报告 fits=False，让引擎在启动时拒绝而不是跑到一半 OOM。
            lay = Layout(resident=0, slots=0, embed_on_cpu=True, head_on_cpu=True,
                         reason="card too small even for streaming skeleton")
            lay.stream_layers = self.n_layers
            lay.fits = self.used_bytes(lay) <= self.vram
            return lay
        c, lay = best
        # 派生量必须在返回前填好：engine 读 stream_layers 来报「几层要流式搬」。
        lay.stream_layers = self.n_layers - int(lay.resident)
        lay.fits = self.used_bytes(lay) <= self.vram
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

    # ---- 运行中调整：唯一的入口 -------------------------------------------
    #
    # 这一个方法吃掉原来的三样东西：
    #
    #   loading.try_grow()            首次 forward 后按实测 free 扩容
    #   autotune.PID / PrefetchController   每步调预取深度
    #   engine._shed_vram()           OOM 后缩
    #
    # 它们各自只看一个量（free / io_ratio / 报错），做局部决策，互相不知道。
    # 现在统一成：**给定当前布局和实测状态，返回新布局（或不动）**。
    #
    # 为什么不是 PID：PID 假设「连续控制、每步调整、代价为零」。而改布局要
    # 重建槽位池 / 搬层权重，代价是秒到分钟级。对离散、高代价的决策用 PID
    # 是方法论错误 —— 应该直接解目标函数，然后比较「迁移代价 vs 收益」。
    def observe(self, cur: Layout, *, free_vram_b: float | None = None,
                measured: dict | None = None, calibrate: dict | None = None
                ) -> tuple[Layout, str]:
        """喂一次运行时状态，返回 (新布局, 原因)。**不迁移时返回 cur 本身。**
        参数：
          cur           当前布局
          free_vram_b   此刻驱动报的显存余量（byte）。给了就用它算可用上限，
                        比启动时的静态预算准 —— 这是 try_grow 的替代。
          measured      实测值，如 {"stream_s":..., "expert_s":..., "compute_s":...}
                        模型预测与实测的偏差可以用来校正带宽。
          calibrate     新的标定值（带宽变了就有）
        """
        if calibrate:
            self.h2d = float(calibrate.get("h2d_bps") or self.h2d)
            self.disk = float(calibrate.get("disk_bps") or self.disk)
            self.disk_rand = float(calibrate.get("disk_rand_bps") or self.disk_rand)
            self.ram = float(calibrate.get("ram_bps") or self.ram)
            self.cal.update(calibrate)
        # 实测和模型差得多，说明带宽估计错了 —— 按比例校正后再解。
        if measured:
            pred = self.cost(cur)
            p_stream = float(measured.get("stream_s") or 0.0)
            p_expert = float(measured.get("expert_s") or 0.0)
            if p_expert > 0.01 and pred.expert_s > 0.01:
                ratio = p_expert / pred.expert_s
                if ratio > 1.5 or ratio < 0.67:
                    # 专家路径慢/快了 ratio 倍：把带宽反向缩放回去
                    self.disk_rand = max(1.0, self.disk_rand / ratio)
                    self.ram = max(1.0, self.ram / ratio)
        # 用实测 free 重解：这是「显存尽可能占满」的正确实现 ——
        # 不是 try_grow 那种「试到失败」，而是让目标函数在真实上限下重算。
        old_vram = self.vram
        if free_vram_b is not None and self.has_gpu:
            # 已占用 + 空闲 = 这块卡能给这个进程的总量
            used = self.used_bytes(cur)
            self.vram = max(used, used + float(free_vram_b))
        try:
            new = self.solve()
        finally:
            self.vram = old_vram
        move, gain = self.should_migrate(cur, new)
        if not move:
            return cur, "keep: gain %.3fs/step below migration cost" % gain
        return new, "migrate: gain %.3fs/step" % gain

