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
from math import ceil

GB = 1 << 30
MB = 1 << 20

# **规划器里不允许常数。**
#
# 这里曾经有三个 _FITS_FRACTION = 0.85 / WS_BASE = 0.20GB / WS_PER_TOKEN = 0.02GB。
# 它们都是某台机器上量出来然后写死的数，注释也承认了来源（实测 7.46/7.93GB）。
#
# 正确来源分别应该是：
#   可用容量  <- 驱动（mem_get_info，在真正分配之前问）
#   workspace <- 引擎实测（模型构建前后 alloc 之差），回填到 calibrate
#
# 拿不到就不解。规划器宁可拒绝，也不拿编造的数字解一个看起来合理的布局。


@dataclass
class Layout:
    """一份布局。目标函数的变量就是这几个。"""

    # resident 是权威字段，engine 通过 gpu_layers 属性读它（同名不同叫法）。
    resident: int = 0
    slots: int = 0
    weight_buffers: int = 0
    weight_buffer_bytes: int = 0
    cache_bytes: int = 0
    host_pool_bytes: int = 0
    host_weight_bytes: int = 0
    host_reserve_bytes: int = 0
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
        self.vram = float(self.cal["usable_vram_bytes"]) if "usable_vram_bytes" in self.cal else sum(c.free_gb for c in caps) * GB
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
        # **没有任何可调量带默认值。**
        #
        # 这里曾经是 `self.compute_per_layer_s = self._tunable(...)` 配一个
        # `_tunable(name, default, cast)` —— 名字叫 tunable，实际语义是
        # 「找不到就退回一个写死的数」。三个默认值（0.0025 秒/层、0.5 的 pin
        # 比例、None 的 workspace）全都是某台机器上量出来的。
        #
        # 现在改成 required：拿不到就抛，让调用方知道必须先测。
        #
        # 注意 resident / slots / cache_bytes 从来不在这个列表里 —— 它们由目标
        # 函数解出来，不是输入。以前把 slots 做成可调导致规划器被上一轮的旧值
        # 永久锁死。
        # **这两个量的来源是「引擎启动时自己测」，不是外部预先准备好。**
        #
        # 之前的 _required 会抛「先跑 tinyinfer.tune」—— 那等于要求人先做一步。
        # 而引擎自己就能测：compute_per_layer_s 是一次前向的耗时除以层数，
        # pin_ratio 是主机内存里能 pin 的比例。让引擎在第一次 resolve() 之前
        # 调 measure()，比要求使用者先跑训练器合理。
        #
        # 仍然保留 calibrate / 环境变量两条覆盖路径（训练器和 A/B 测试要用），
        # 但**没有默认值** —— 三条来源全都没有就抛。
        # **惰性读取，不在构造时抛。**
        #
        # 这两个量在「解布局」那一步都不是必需的（compute_per_layer_s 对所有
        # 布局加同一个常数，pin_ratio 只影响缓存里 pin 多少）。构造时就要求它们
        # 存在，会让引擎在「还没跑前向」时直接起不来。
        #
        # 真正用到的那一刻（cost() / used_bytes()）才读，那时引擎已经能测了。
        self._compute_per_layer_s_raw = None
        self._pin_ratio_raw = None


    @property
    def compute_per_layer_s(self):
        return self._required("compute_per_layer_s", float)

    @property
    def pin_ratio(self):
        return self._required("pin_ratio", float)

    def measure(self) -> dict:
        """引擎启动时自己测出解布局所需的全部实测量。

        这些量原来都写死成常数（0.0025 秒/层、0.5 的 pin 比例、0.3 秒的
        graph 收益……）。它们**不由「资源量 ÷ 单元大小」推出**，所以只能测。

        但「只能测」不等于「必须由外部先测」：引擎自己就有全部条件 ——
        它知道模型有多少层、能跑一次前向、能问驱动、能读系统内存。

        这个方法在 resolve() 之前被调用一次，返回测到的值，同时写进 self.cal。
        """
        import time as _time

        out: dict = {}
        # 1) 每层计算耗时 = 一次前向 / 层数。需要调用方已把模型放到 device 上。
        _bench = getattr(self, "_bench_forward", None)
        if _bench is not None:
            _n = max(1, int(self.n_layers))
            _t0 = _time.perf_counter()
            _bench()
            if self.has_gpu:
                import torch

                torch.cuda.synchronize()
            out["compute_per_layer_s"] = (_time.perf_counter() - _t0) / _n

        # 2) pin 比例 = 主机内存里允许 pin 的比例。由预算器给，不是常数。
        try:
            from .budget import budget as _hb

            _room = float(_hb().room())
            _total = float(getattr(_hb(), "total", 0) or 0)
            if _total > 0:
                out["pin_ratio"] = max(0.0, min(1.0, _room / _total))
        except Exception:
            pass

        # 3) 本机实测的搬运耗时（引擎在第一次搬层时记录）。
        _h2d = getattr(self, "measured_h2d_layer_s", None)
        if _h2d:
            out["h2d_layer_s"] = float(_h2d)
        _hex = getattr(self, "measured_h2d_expert_s", None)
        if _hex:
            out["h2d_expert_s"] = float(_hex)

        self.cal.update(out)
        return out
    def _required(self, name, cast):
        """读一个**必须存在**的实测值。缺失就抛，没有默认值。

        """
        import os as _os

        env_key = "TI_TUNE_" + name.upper()
        v = _os.environ.get(env_key)
        if v not in (None, ""):
            return cast(v)
        v = self.cal.get(name)
        if v is not None:
            return cast(v)
        raise RuntimeError(
            f"Planner: {name} has not been measured. The planner does not carry "
            "defaults — run tinyinfer.tune against this model/machine first, or "
            "pass it via calibrate.")

    # ---- 目标函数 ---------------------------------------------------------

    def _layer_bytes(self) -> tuple[int, ...]:
        sizes = self.model.get("layer_bytes_by_id")
        if not isinstance(sizes, dict):
            raise RuntimeError("planning requires per-layer byte sizes from checkpoint metadata")
        try:
            values = tuple(ceil(sizes[i] if i in sizes else sizes[str(i)])
                           for i in range(self.n_layers))
        except KeyError as exc:
            raise RuntimeError("planning is missing a layer byte size") from exc
        if any(size <= 0 for size in values):
            raise ValueError("layer byte sizes must be positive")
        return values

    def _weight_stream_seconds(self, resident: int, buffers: int) -> float:
        sizes = self._layer_bytes()[resident:]
        if not sizes:
            return 0.0
        transfer = self._required("h2d_layer_s", float) / self.layer_b
        compute = self.cal.get("compute_per_layer_s")
        if compute is None or buffers <= 1:
            return sum(sizes) * transfer
        # 模拟串行复制流与计算流；同一环槽只能在上一使用者计算结束后复用。
        released = [0.0] * buffers
        copied = computed = 0.0
        for step, size in enumerate(sizes):
            slot = step % buffers
            copied = max(copied, released[slot]) + size * transfer
            computed = max(computed, copied) + float(compute)
            released[slot] = computed
        return computed - len(sizes) * float(compute)

    def cost(self, lay: Layout) -> Cost:
        """估算这份布局的一步耗时。**唯一的判据。**"""
        n = self.n_layers
        stream = max(0, n - lay.resident)
        # **用「一层实测搬多久」而不是「字节 / 带宽」。**
        #
        # 字节除带宽算出来偏乐观：474.8MB / 7.2GB/s = 66ms，而实测 b.to(cuda)
        # 搬一层要 141ms（3.4GB/s 有效带宽）—— 差 2.1 倍。
        #
        # 原因是带宽测的是「一块连续内存的 DMA」，而搬一层是 30+ 个张量、
        # 每个都要 set_weight、可能有多个 stream 同步。那个开销不在带宽里。
        #
        # **用 h2d_layer_s（实测「搬一层要几秒」），必须有值。**
        #
        # 曾经写成 `layer_b / h2d * 2.1` —— 那个 2.1 是「连续 DMA 带宽」和
        # 「一层 30 多个张量各自 set_weight」之间的实测差距。它随模型结构变
        # （张量个数不同）、随驱动变，写死就是把一台机器的比值当定律。
        stream_s = self._weight_stream_seconds(lay.resident, lay.weight_buffers)

        # **一步要几个不同的 (layer, expert)，以及「下一步还要重搬多少」。**
        #
        # 原来写的是 uncovered = need - slots，slots >= need 时 expert_s 直接
        # 归零 —— 规划器于是认为「槽位覆盖整步需求 = 专家零成本」，拼命加槽位
        # （本机解出 245 个、占 4.5GB 显存 + 4.5GB 主机预算，把常驻层和缓存
        # 都挤掉了）。
        #
        # 但路由**每一步都在变**：实测把 240 个需求全装进槽位之后，相邻两步
        # 的 (layer, expert) 集合只重合 35%。也就是说每步仍有约 65% 要换进来。
        # 换的时候字节从哪来，完全取决于主机缓存：
        #   cache 够大 -> 从内存搬到显存（ram 带宽）
        #   cache 不够 -> 读盘（disk 带宽）
        # 这才是槽位和缓存真正联动的地方，也是 T 里唯一能同时反映两者的项。
        need = self.topk * n
        # 相邻两步的路由重合度。**实测 0.988，不是 0.35。**
        #
        # 0.35 那个数是拿 5 个 token 一起跑出来的（那是 prefill 形状，
        # _last_ids 每层有 12~28 个专家），不能当 decode 的单步复用率。
        # 单 token decode 的实测快照：
        #   步后槽位 242，保留 239，新增 3  ->  保留率 98.8%
        # 也就是说槽位覆盖整步之后，每步只有两三个专家要换。
        #
        # 这个数字直接决定「槽位 vs 常驻层」的取舍：按 0.35 算，每步要搬
        # 156 个专家、expert_s=1.85s，规划器就会得出「槽位无用」的结论；
        # 按 0.988 算，每步只搬 3 个、expert_s 接近 0，槽位才显得值那 4.5GB。
        # **实测值。** 它由这份 checkpoint 的路由分布决定，换模型必变。
        #
        # 但它在「解布局」这一刻还测不到 —— 要跑两步才知道相邻 token 的路由
        # 重合多少。所以这里允许「尚未测到」：此时按 0 处理（槽位不留任何
        # 复用价值，规划器因此不会为了槽位牺牲常驻层），等引擎跑出第一步后
        # 回填，下次 observe() 重解时就用真实值。
        #
        # 这不是兜底：0 是一个明确的「未知」语义，而且它的影响是保守方向
        # （宁可少给槽位），不是编造一个看起来合理的数。
        # **未知时不能用 0。**
        #
        # 我一开始把「未测到」按 0 处理，理由是「保守：认为槽位无复用价值，
        # 于是不会为了槽位牺牲常驻层」。方向错了 —— 槽位在这台机器上是**可用性
        # 前提**：slots=0 时 MoE 连纯 GPU 路径都进不去（
        #   arena unusable: 0 blocks, 0 buffers），整个模型跑不起来。
        #
        # 所以「未知」的正确处理是：按「覆盖不到整步」的最坏情况给槽位（重用率
        # 取 0 会让求解器完全放弃槽位，那是另一个极端）。这里取一个明确的
        # 「不知道就先按 2 步换一遍」的中点，等引擎跑出真实值后 observe() 覆盖。
        try:
            _reuse = float(self._required("route_reuse", float))
        except RuntimeError:
            # 未知：假设一半专家会换（0.5），既不给零槽位、也不假装全命中
            _reuse = 0.5
        _reuse = min(1.0, max(0.0, _reuse))
        # **本步要从「槽位之外」装进显存的专家数。**
        #
        # 一步一步看清楚：本步需要 need 个 (layer, expert)。它们中的 1-reuse
        # 是新的（上一槽位里没有），必须搬进来；复用的那部分已经在显存里。
        #
        # 关键：**搬进来的这些能不能留下，取决于槽位够不够。** 槽位 >= need
        # 时，本步的 need 个和上一步的 need 个能同时驻留，下一步的复用率就是
        # 真实的 0.35 —— 也就是说只有 (1-reuse)*need 需要新搬。
        # 槽位 < need 时，池子装不下一步的全部需求，上一步的专家被本步挤掉，
        # 下一步又得全搬 —— 命中率塌到 0。
        #
        # 所以 effective_reuse 不是常数，它由「槽位 / need」这个覆盖比决定：
        #   覆盖比 = 1  -> 复用率 = route_reuse
        #   覆盖比 < 1  -> 复用率线性衰减（槽位空着的地方留不住东西）
        _cover = min(1.0, float(max(0, int(lay.slots))) / max(1.0, float(need)))
        _eff_reuse = _reuse * _cover
        uncovered = need * (1.0 - _eff_reuse)
        if self.expert_b <= 0 or uncovered <= 0:
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
            # **专家搬运的耗时必须实测，不能用字节 / 带宽估。**
            #
            # 估出来是 18.8MB / 35GB/s = 0.54ms/个，而实测（含 slot 查表、
            # set_weight、event 同步）大约 1.5-2ms/个。差 3 倍。
            #
            # 这个系数直接决定「槽位 vs 常驻层」的取舍：按 0.54ms 算，244 个
            # 槽位省 130ms 只花 4.5GB；按 1.8ms 算，244 个槽位省 430ms —— 但
            # 那 4.5GB 挤掉 9 层常驻，而 9 层 x 141ms = 1.27s。**槽位仍然不划算**，
            # 而且要差 3 倍才看得出来。
            # **实测值，无默认。** 它含 slot 查表、set_weight、event 同步，
            # 随槽位池实现和驱动变。
            _per_exp_s = self._required("h2d_expert_s", float)
            expert_s = uncovered * (float(_per_exp_s) + (0.0 if hit > 0.5
                                                          else self.expert_b / max(1.0, self.disk_rand)))

        # 计算：每步都要过全部层，与布局无关。用实测的每层计算耗时。
        # **每层计算耗时是「每层都要过一遍」的固定开销，与布局无关。**
        #
        # 换句话说：它对所有候选布局加的是同一个常数，不改变「哪个布局更好」。
        # 所以它不必在解布局之前就存在 —— 没有它就用 0（T 少算这一项，但排序
        # 不变），有它就加上。这样引擎不会因为「还没跑过一次前向」就起不来。
        try:
            compute_s = float(self._required("compute_per_layer_s", float)) * n
        except RuntimeError:
            compute_s = 0.0

        # **全常驻解锁 CUDA graph，这是目标函数里之前漏掉的一整块。**
        #
        # engine._graph_capture_shape 有一条硬条件：`gpu_layers < max_layers`
        # 就直接返回 False。也就是说只要有层在流式，整个 CUDA graph 都用不了，
        # 每步都走 eager：40 层 x 几十个 Python kernel 调用、每次 forward 重新
        # 调度、wall 和 CUDA 汇总之间那 300ms 的空转全在这里。
        #
        # 实测（单 token，wall=3065ms）：
        #   CUDA 汇总 2750ms，空转 ~300ms（graph 关闭时的调度开销）
        # 全常驻时这 300ms 归零，而且 graph 还能消掉 kernel 之间的 gap。
        #
        # 所以全常驻的收益不只是「省掉流式搬运」，还有「解锁 graph」。
        # 这一项不进目标函数的话，规划器永远选不到全常驻 —— 而它本该在
        # 「搬 33 层花 0.94s」和「全常驻 + graph」之间做正确取舍。
        # **全常驻解锁 CUDA graph 的收益，必须实测。**
        #
        # 这里曾经写 `graph_gain_s = 0.30` —— 一个「graph 能省 0.3 秒」的
        # 固定收益。它随模型深度、kernel 数量、主机速度变，写死等于假设
        # 所有模型都能省同样多。
        #
        # 正确来源：同一份布局分别在 graph / eager 下各跑一次，差值就是它。
        # 没测过就当作 0（保守：不影响布局选择），但绝不编一个数。
        if lay.resident >= n:
            try:
                graph_gain = float(self._required("graph_gain_s", float))
            except RuntimeError:
                graph_gain = 0.0
            compute_s = max(0.0, compute_s - graph_gain)
        return Cost(stream_s=stream_s, expert_s=expert_s, compute_s=compute_s)

    # ---- 资源约束 ---------------------------------------------------------

    def used_bytes(self, lay: Layout) -> float:
        """这份布局在显存里占多少。"""
        b = 0.0
        # **每层权重按实际字节计，不乘 slack。**
        #
        # 原来乘 1.7（"权重 + activation/frag"），但实测：搬一层 474.8MB 的权重
        # 进显存，memory_allocated 正好增加 474.8MB —— 倍数是 1.00，没有额外开销。
        #
        # 那个 1.7 是旧规划器的实测常数（它把 activation 和碎片算进每层），
        # 但那部分已经由 runtime_b / ws0 单独记了，再乘一次就是重复扣减。
        #
        # 后果很直接：按 1.7 算，475MB/层变成 808MB/层，8GB 卡只给 4 层；
        # 按实际算能放 10 层。README 记录的历史成绩（1.26 tok/s）正是 10 层那档。
        if not 0 <= lay.resident <= self.n_layers:
            raise ValueError("resident layer count is outside the model")
        b += sum(self._layer_bytes()[:lay.resident])
        b += lay.slots * self.expert_b
        if not lay.embed_on_cpu:
            b += self.embed_b
        if not lay.head_on_cpu and not self.tied_head:
            b += self.head_b
        if lay.weight_buffers < 0 or lay.weight_buffer_bytes < 0:
            raise ValueError("weight ring capacity cannot be negative")
        b += lay.weight_buffers * lay.weight_buffer_bytes
        # **workspace 由引擎实测回填，规划器不猜。**
        #
        # 这里曾经是 `b += 0.20 * GB` —— 一个「cuBLAS handle + 分配器碎片大约
        # 200MB」的估值。它随 kernel 的 tile 形状、cuBLAS 版本、分配器状态变。
        #
        # 来源应该是：引擎构建完、第一次 forward 前后各读一次 memory_allocated，
        # 差值就是真实的运行时开销，回填到 calib['runtime_overhead_bytes']。
        _rt = self.cal.get("runtime_overhead_bytes")
        if _rt is None:
            raise RuntimeError(
                "Planner.used_bytes: runtime overhead (workspace, cuBLAS handles, "
                "allocator slack) has not been measured for this engine build. "
                "It must be back-filled by the engine; there is no constant.")
        b += float(_rt)
        # **运行时固定开销：rope 表。**
        #
        # 这是模型常量（1M 位置 x rotary_dim），不是每层权重，但它必须驻留
        # 在显存上才能算 attention。本机实测两张表各 268.4MB，合计 0.54GB。
        #
        # 规划器原来完全不算这一项，于是解出来的布局在「纸面上」装得下、
        # 实际跑到 rope_cache.get_table 就在 t.to(device) 那里 OOM：
        #   RuntimeError: !handles_.at(i) INTERNAL ASSERT FAILED
        # 这类 OOM 不该在运行期暴露 —— 规划器要么把它算进预算，要么明确
        # 说「这张卡放不下这个模型」。
        #
        # **rope 表的字节由配置算，算不出来就抛。**
        #
        # 曾经写成「按 seq_len 和 rotary_dim 算，拿不到配置时按 0.54GB 兜底」——
        # 那个 0.54GB 是本机实测的两张表之和（各 268.4MB），给别的模型完全不成立。
        _seqlen = int(self.model.get("max_position_embeddings") or 0)
        _rot = int(self.model.get("rotary_dim") or 0)
        if _seqlen <= 0 or _rot <= 0:
            raise RuntimeError(
                "Planner.used_bytes: rope table size unknown "
                f"(max_position_embeddings={_seqlen}, rotary_dim={_rot}). "
                "It must come from the model config; there is no fallback size.")
        # 两张表（yarn 与原始各一），complex64 = 8 字节/元素
        b += 2 * _seqlen * (_rot // 2) * 8
        return b

    def fits(self, lay: Layout) -> bool:
        """这份布局装得下吗。**口径必须和真实分配一致。**

        这里曾经是 `used_bytes(lay) <= vram * 0.85` —— 一个「留 15% 给运行时
        开销」的比例。实测它不够也不统一：同样的 0.85 在不同模型上留出的绝对
        字节数不同，而运行时开销（rope 表、KV、activation、分配器碎片）不是
        按比例长的。

        现在按三件事分别判，每一件都有明确来源：

          used_bytes  <= usable_vram - headroom

        其中 headroom = 最大单次分配的字节数。为什么需要它：rope 表是
        268MB 的**单个张量**，分配时要一块连续的显存；而 used_bytes 只保证
        「稳定态装得下」。实测 6.81GB / 6.86GB（差 61MB）时，rope 那 268MB
        照样 OOM —— 那是「总账够、单笔不够」。
        """
        _headroom = float(self.cal.get("max_single_alloc_bytes") or 0.0)
        if _headroom <= 0:
            # 没测过最大单次分配时，用「这份布局真正要放到显存上的最大张量」
            # 当代表 —— 从 header 读出来，不是常数。
            #
            # 注意只算**实际进显存**的：embed/head 放在 CPU 时（embed_on_cpu /
            # head_on_cpu）它们不参与显存分配，拿它们当 headroom 会把预算砍掉
            # 1.3GB（实测：headroom 取 embed_b=1324MB 后 resident 从 5 掉到 0）。
            _cand = [self.max_layer_b]
            if not lay.embed_on_cpu:
                _cand.append(self.embed_b)
            if not lay.head_on_cpu and not self.tied_head:
                _cand.append(self.head_b)
            _headroom = max(_cand)
        streamed = self.n_layers - lay.resident
        if streamed:
            if not 1 <= lay.weight_buffers <= streamed:
                return False
            if lay.weight_buffer_bytes < max(self._layer_bytes()[lay.resident:]):
                return False
        elif lay.weight_buffers or lay.weight_buffer_bytes:
            return False
        return (self.used_bytes(lay) + _headroom) <= self.usable_vram()

    def usable_vram(self) -> float:
        """本次规划的容量快照；求解过程中不读取另一台机器的 CUDA 状态。"""
        return self.vram

    def _plan_host_pool(self, lay: Layout) -> bool:
        """从逐层字节和物理余量分配权重源、打包周转与专家缓存。"""
        layer_bytes = self._layer_bytes()
        # 规划只用这一个余量快照；引擎用同一快照分配，不会因再采样而拒绝。
        available = int(self.model["host_available_bytes"])
        if available <= 0:
            return False
        lay.host_weight_bytes = sum(layer_bytes[lay.resident:])
        # 全局权重不从层池中分配；逐层构建只保留一个转换工作区。
        globals_bytes = ceil(self.embed_b + (0 if self.tied_head else self.head_b))
        workspace = ceil(self.max_layer_b)
        lay.host_reserve_bytes = globals_bytes + workspace
        # 打包当前层时，原始权重与目标 slab 会同时存在。
        pool_required = lay.host_weight_bytes + workspace
        cache_room = available - lay.host_reserve_bytes - pool_required
        if cache_room < 0:
            return False
        cache_room -= cache_room % 256  # 下一个权重 slab 的 ABI 对齐。
        if self.expert_b > 0:
            total_experts = int(self.model["n_experts"]) * self.n_layers
            entries = min(total_experts, int(cache_room // self.expert_b))
            lay.cache_bytes = int(entries * self.expert_b)
        else:
            lay.cache_bytes = 0
        lay.host_pool_bytes = pool_required + (lay.cache_bytes + 255) // 256 * 256
        return True

    # ---- 求解 -------------------------------------------------------------

    def solve(self) -> Layout:
        """共同求解常驻层、专家槽位和权重环，不允许引擎另定容量。"""
        if not self.has_gpu:
            return Layout(resident=0, slots=0, stream_layers=self.n_layers,
                          embed_on_cpu=True, head_on_cpu=True,
                          reason="no GPU: weights stay on host")
        best = None
        capacity = self.usable_vram()
        for resident in range(self.n_layers, -1, -1):
            streamed = self.n_layers - resident
            stream_sizes = self._layer_bytes()[resident:]
            buffer_counts = range(1, streamed + 1) if streamed else (0,)
            uniform_sizes = len(set(stream_sizes)) <= 1
            previous_stall = None
            for buffers in buffer_counts:
                stall = self._weight_stream_seconds(resident, buffers)
                # 相同流水线代价下，更多环槽只增加内存需求，候选严格被支配。
                if uniform_sizes and previous_stall is not None and stall == previous_stall:
                    break
                previous_stall = stall
                buffer_bytes = max(stream_sizes) if buffers else 0
                for embed_cpu, head_cpu in ((False, False), (True, False), (True, True)):
                    probe = Layout(resident=resident, slots=0,
                                   weight_buffers=buffers, weight_buffer_bytes=buffer_bytes,
                                   embed_on_cpu=embed_cpu, head_on_cpu=head_cpu)
                    free = capacity - self.used_bytes(probe)
                    if free < 0:
                        continue
                    if self.expert_b > 0:
                        # 纯 GPU MoE 至少容纳一次真实路由；多余槽位不能创造复用。
                        max_slots = min(int(free // self.expert_b), self.topk * self.n_layers)
                        slot_counts = range(self.topk, max_slots + 1)
                    else:
                        slot_counts = (0,)
                    for slots in slot_counts:
                        lay = Layout(resident=resident, slots=slots,
                                     weight_buffers=buffers, weight_buffer_bytes=buffer_bytes,
                                     stream_layers=streamed,
                                     embed_on_cpu=embed_cpu, head_on_cpu=head_cpu)
                        if not self._plan_host_pool(lay):
                            continue
                        if not self.fits(lay):
                            continue
                        cost = self.cost(lay)
                        rank = (cost.total_s, self.used_bytes(lay), buffers)
                        if best is None or rank < best[0]:
                            best = (rank, cost, lay)
        if best is None:
            raise RuntimeError("no feasible layout for the measured memory budget")
        _, cost, lay = best
        lay.reason = ("T=%.3fs (stream %.3f + expert %.3f + compute %.3f); weight ring %dx%d bytes"
                      % (cost.total_s, cost.stream_s, cost.expert_s, cost.compute_s,
                         lay.weight_buffers, lay.weight_buffer_bytes))
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

