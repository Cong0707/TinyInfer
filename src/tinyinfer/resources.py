"""Hardware detection + automatic resource planning for tinyinfer.

Everything that used to be a hand-tuned constant (resident gpu_layers, expert slot
count, RAM cache size, prefill chunk) is derived here from the machine the engine
runs on, so tinyinfer adapts to 8GB laptops, 24GB desktops and unified-memory boxes
without manual flags.

Calibration (measured on RTX 5060 8GB Laptop + 47GB RAM):
  * one resident fp8 layer  : ~0.44 GB VRAM (weights 0.19 + allocator/state)
  * embed (129280x5120 bf16) : 1.32 GB VRAM
  * head  (same shape)       : 1.32 GB VRAM (can stay on CPU)
  * one expert slot          : 18.8 MB VRAM (48 slots ~= one decode token's experts)
  * one streamed layer RVAM : ~0.19 GB host RAM (weights) + experts in the RAM cache
  * practical VRAM ceiling   : physical - 0.6 GB (display/context/FX) - 0.8 GB headroom
"""
from __future__ import annotations

import dataclasses
import os

GB = 1 << 30
MB = 1 << 20

# calibrated model-side constants (fp8 non-expert weights, bf16 embed/head)
LAYER_GB = 0.55           # resident per-layer VRAM incl. allocator/state (recalibrated:
                          # 11 layers + embed + slots measured 8671MB, i.e. 0.55GB/layer)
LAYER_HOST_GB = 0.47      # per-layer host RAM for a streamed layer (measured: the
                          # 34 streamed layers need a 15.9GB pinned slab, 468MB each)
EMBED_GB = 1.32
HEAD_GB = 1.32
EXPERT_MB = 18.8
KV_GB_PER_KTOK = 0.02     # compressed+window KV for 40 layers per 1k tokens (rough)


@dataclasses.dataclass
class Card:
    """One CUDA device as the planner sees it."""
    index: int
    name: str
    total_gb: float
    free_gb: float


@dataclasses.dataclass
class Hardware:
    name: str
    vram_gb: float
    ram_gb: float
    ram_avail_gb: float
    unified: bool = False
    cards: list[Card] = dataclasses.field(default_factory=list)

    @property
    def vram_budget_gb(self) -> float:
        """Total VRAM that may be used across every card."""
        # Use physical capacity, not the snapshot of "free": a leftover CUDA context or
        # the desktop compositor makes free look 1GB smaller and the planner then
        # under-fills the card (0 resident layers on an 8GB card that comfortably runs 2).
        vram = sum(c.total_gb for c in self.cards) if self.cards else self.vram_gb
        if self.unified:
            # One physical pool: the driver's wired-memory limit is ~2/3 of RAM on
            # Apple silicon, and counting RAM *and* VRAM separately double-books the
            # same bytes (a 48GB Mac planned 46GB of VRAM plus 36GB of cache).
            vram = min(vram if vram > 0 else self.ram_gb, self.ram_gb * 0.70)
        # Per-card reserve: display/context/FX, allocator slack and the transient
        # buffers one layer needs while it is being copied in. 0.9GB was measured to be
        # too tight (a planned 6.4GB layout actually allocated 7.0GB on an 8GB card).
        return max(0.0, vram - 1.5 * max(1, len(self.cards) or 1))

    @property
    def ram_budget_gb(self) -> float:
        return max(0.0, self.ram_avail_gb - 4.0)  # leave 4GB for the OS/process


def detect() -> Hardware:
    """Detect the machine and return a Hardware descriptor (never raises).

    All CUDA devices are enumerated: the planner spreads layers over every one of
    them. A device is skipped when it cannot hold the smallest useful unit (one
    layer buffer + its share of slots), e.g. a 2GB card next to an RTX 6000.
    """
    vram_gb = 0.0
    name = "cpu"
    cards: list[Card] = []
    try:
        import torch

        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()   # a leftover context must not shrink the plan
            except Exception:
                pass
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                try:
                    free, _total = torch.cuda.mem_get_info(i)
                    free_gb = free / GB
                except Exception:
                    free_gb = props.total_memory / GB
                cards.append(Card(index=i, name=props.name,
                                  total_gb=props.total_memory / GB, free_gb=free_gb))
            vram_gb = sum(c.free_gb for c in cards)
            if cards:
                name = cards[0].name if len(cards) == 1 else f"{len(cards)}x {cards[0].name}"
    except Exception:
        cards = []
    ram_gb = ram_avail = 0.0
    try:
        import psutil

        vm = psutil.virtual_memory()
        ram_gb = vm.total / GB
        ram_avail = vm.available / GB
    except Exception:
        pass
    return Hardware(name=name, vram_gb=vram_gb, ram_gb=ram_gb, ram_avail_gb=ram_avail,
                    cards=cards)


@dataclasses.dataclass
class Plan:
    gpu_layers: int
    head_on_cpu: bool
    expert_slots: int
    cache_bytes: int
    prefill_chunk: int
    # 会被 streaming 的层数 = n_layers - resident。**没有消费者**：
    # engine 用 _stream_layers() 从 gpu_layers 自己推，不看这个字段。
    # 保留它是给 reason 文本和外部观察用的。
    stream_layers: int
    reason: str
    # False when even the streaming skeleton (ring + workspace + tables) does not fit
    # the VRAM budget: the engine refuses to start instead of OOMing mid-prefill.
    fits: bool = True
    # ---- 主机侧的统一账面 --------------------------------------------------
    #
    # 下面四个字段存在的原因，是这几个数以前散在 engine.py / budget.py / loading.py
    # 里各算各的，而且都写死了跟「这台机器」有关的常数（6GB 余量、240MB/层、
    # 1.15 系数、16 线程）。换一台机器那些数就不对，而且看代码看不出来它们
    # 互相约束。统一在这里算好，engine 只读字段。
    #
    # 规划器输出的是「计划」，不是「保证」：真实可用内存可能在 plan 之后变化，
    # 所以 engine 拿到这些数仍然要按实际 free 收缩（claim 失败就缩），
    # 但**收缩的起点是这里给的数**，不是别处另算一个。
    #
    # 宿主池的目标字节数（专家缓存 + slab + 构建期临时都在里面）。
    host_pool_bytes: int = 0

# 注：这里原来还有 slab_reserve_bytes / pin_budget_bytes 两个字段，但没人读 ——
# engine 里的 slab 预留是它自己按 pool 现算的，pin 预算直接用 pin_ratio。
# test_plan_wiring.py 会抓这种「Plan 里有、但没人消费」的字段，正是它报出来的。
    # 缓存里 pin 的比例。这个数**没有理论值**，只能实测：pin 得越多，专家上显存
    # 走 async DMA 的机会越大；但 grant() 会记账进 host budget，pin 多了反而
    # 压缩可用缓存。训练器（tinyinfer.tune）在本机扫出来 0.3/0.5/0.7/1.0 分别对应
    # 2880/2700/2696/2742ms，最优点在 0.7。tuned.json 里存的就是它。
    pin_ratio: float = 0.5
    # 专家读取的并发深度。这是 SSD 队列深度（QD 16 就饱和），不是 CPU 数，
    # 所以它跟机器大小无关 —— 但仍然放在这里，因为它是规划的一部分：
    # 并发深度决定「一笔 decode 的 miss 需要多久」。
    read_threads: int = 16
    # layer -> CUDA index, and per-card expert slot counts. Single-GPU boxes get
    # {layer: 0} and {0: slots} and the engine behaves exactly as before.
    # 每层放在哪张卡。**没有消费者**：多卡时层放置由 engine 自己的
    # _layer_dev 决定，这个字段只在 plan 内部用来算 card_slots。
    # 和 card_slots 一样，不要以为它生效。
    layer_card: dict[int, int] = dataclasses.field(default_factory=dict)
    # 每张卡分到多少专家槽位。**目前没有消费者**：decode 的槽位分配由
    # VramSlotCache.set_devices 按设备实际余量自己做，不读这个字段。
    # 它是给「想按计划预先分卡」的调用方留的接口（例如启动时要打印/校验），
    # 不要以为它是生效的分配 —— 和 _expert_dev 一样容易看错。
    card_slots: dict[int, int] = dataclasses.field(default_factory=dict)
    embed_on_cpu: bool = False   # 1.32GB of VRAM for a 10KB row lookup: waste when tight
    speculation: bool = False    # the plan actually keeps room for MTP draft stages


def _layer_of(name: str) -> int | None:
    """The layer index a tensor belongs to, whatever the checkpoint's prefix is.

    DeepSeek names layers "layers.N.attn..."; Qwen nests them as
    "model.language_model.layers.N.self_attn...". Both contain a "layers.<int>" segment,
    so find that rather than assuming a prefix.
    """
    parts = name.split(".")
    for i, part in enumerate(parts):
        if part == "layers" and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                return None
    return None


def model_profile(index: Any) -> dict:
    """Measure what the checkpoint actually costs, per layer and per expert.

    Everything the planner needs comes from here, so it adapts to ANY checkpoint
    instead of a hand-tuned table for this one: layer bytes (non-expert), the largest
    layer (the streaming buffer size), the embedding and head tables, and one expert
    blob. Falls back to calibrated constants only if the index cannot be read.
    """
    prof = {"layer_bytes": 0.19 * GB, "max_layer_bytes": 0.72 * GB,
            "embed_bytes": 1.32 * GB, "head_bytes": 1.32 * GB,
            "expert_bytes": 18.8 * MB, "engram_bytes": 2 * GB, "mtp_bytes": 0, "n_layers": 0}
    try:
        wm = getattr(index, "weight_map", None) or {}
        header = getattr(index, "_header", None)
        if not wm or header is None:
            return prof
        per_layer: dict[int, int] = {}
        for name in wm:
            # Layer tensors are "the ones a layer needs per token". Rather than hardcode a
            # naming scheme (this used to check for attn./ffn.gate./hc_, which is
            # DeepSeek's layout and matched nothing at all in a Qwen checkpoint, so every
            # Qwen layer was costed at DeepSeek's 190MB), derive it from the layer index
            # the checkpoint itself declares.
            lid = _layer_of(name)
            if lid is None:
                continue
            parts = name.split(".")
            if "experts" in parts:
                continue          # routed experts are counted per blob, not per layer
            if "engram" in parts:
                # engram.embed.weight/scale are the giant per-engram-layer lookup tables
                # (91.5GB each on DeepSeek-V4.1-Flash). They are row-read from disk and
                # never mapped into a layer buffer, so charging them to the layer made
                # max_layer_bytes 94.7GB and the plan conclude the streaming skeleton
                # needed 192GB on an 8GB card. Only the wkv projection is resident, and
                # it is already counted into engram_bytes below.
                continue
            # Everything else that belongs to a layer IS what the layer streams: the
            # engine asks the layer for its weight paths, so the planner must count the
            # same set. This used to filter on DeepSeek's tensor names (attn./ffn./hc_),
            # which matched nothing in a Qwen checkpoint -- every Qwen layer was costed at
            # DeepSeek's 190MB and the plan then streamed a model that fits entirely.
            try:
                # _header returns (tensor_dict, data_offset), and it raises when a shard is
                # incomplete. Treating the tuple as the dict meant every lookup failed and
                # the whole profile silently stayed at the calibrated defaults -- which
                # happened to be about right for DeepSeek and wildly wrong for anything
                # else (a Qwen layer is ~5MB, not 190MB).
                parsed = header(wm[name])
                tensors = parsed[0] if isinstance(parsed, tuple) else parsed
                ent = tensors.get(name)
                if ent is None:
                    continue
                n = ent["data_offsets"][1] - ent["data_offsets"][0]
            except Exception:
                continue
            per_layer[lid] = per_layer.get(lid, 0) + int(n)
        if per_layer:
            prof["n_layers"] = max(per_layer) + 1
            # **streamed 层的实际搬运量是 header 字节的 2.7 倍，不是 1.0 倍。**
            #
            # header 里记的是 fp8 载荷（每层约 166MB）。但一层真正拷进显存环形缓冲
            # 的是「一块连续 buffer」，每块按最坏层开 463MB（实测），而规划器这里
            # 只按 1.0x 算成 173MB —— 三个环就差 0.87GB。
            #
            # 实测：8GB 卡上规划器给 11 层常驻 + 3 个环，Engine 构建完只剩 1.90GB，
            # 第一次 forward 就在 layer 12 上 OOM（vram free=0MB）。
            #
            # 2.7 这个系数来自两条独立测量：本机 463MB / 173MB = 2.68；
            # 以及下面 plan() 里注释记的「719MB vs 355MB」= 2.03（那台机器 header
            # 更小）。取 2.7 是保守的一侧 —— 显存算少了会 OOM，算多了只是少几个常驻层。
            _m = 2.7
            avg = _m * sum(per_layer.values()) / len(per_layer)
            mx = _m * max(per_layer.values())
            # Two different numbers, used for two different things:
            #   layer_bytes     -> average, for "how many layers fit" (resident * avg)
            #   max_layer_bytes -> worst case, for sizing one flat/prefetch buffer
            # This used to be max(avg, mx), which is just mx: every layer was costed at
            # the worst one. On Qwen3.8-27B (mean 737MB, max 1441MB) that turned a 46GB
            # model into a 92GB plan, so the solver concluded only one layer could be
            # resident, gave the host cache its 256MB floor, and every token then read
            # all 63 streamed layers from disk.
            # layer_bytes 走原始字节（常驻层在显存里就是这些 fp8 载荷，没有环的开销）；
            # max_layer_bytes 走放大后的值（streaming 环按最坏层开一块连续 buffer）。
            prof["layer_bytes"] = sum(per_layer.values()) / len(per_layer)
            prof["max_layer_bytes"] = mx
            # routing width: needed to size the expert pool / decide on coverage
            try:
                import json as _json
                import pathlib as _path
                _cfg = _path.Path(getattr(index, "model_dir", "")) / "config.json"
                if _cfg.exists():
                    # num_experts_per_tok lives under text_config on this checkpoint family
                    # (the top level holds architectures/quantization_config/text_config/...).
                    # Reading only the top level returned None here, which made
                    # has_experts False below and drove the plan to slots=0 -- the one
                    # state that comment at the slots block calls out as fatal: "a zero-slot
                    # plan allocates a fresh 18.8MB buffer per expert per layer per token".
                    # Measured with slots=0: 660 expert visits per token, 100% cache hits,
                    # zero disk reads, and still 1.60s per token, because every visit was a
                    # fresh device allocation instead of a staging slot. Fall back to the
                    # top level so a flat config still works.
                    _raw = _json.loads(_cfg.read_text(encoding="utf-8"))
                    _tc = _raw.get("text_config")
                    _top = _tc if isinstance(_tc, dict) else _raw
                    prof["topk"] = int(_top.get("num_experts_per_tok") or 0) or None
            except Exception:
                prof["topk"] = None
        for key, name in (("embed_bytes", "embed.weight"), ("head_bytes", "head.weight")):
            try:
                h = header(wm[name])
                enc = h[name]
                prof[key] = int(enc["data_offsets"][1] - enc["data_offsets"][0])
            except Exception:
                pass
        # engram projections are loaded on demand and live beside the layer buffers for
        # the rest of the run; not accounting for them is what made an 8GB card OOM in
        # engram.forward on an otherwise valid plan.
        try:
            eng = 0
            for name in wm:
                # only the small wkv projection stays resident; the giant embedding
                # tables (embed.weight/scale) are row-read from disk and never mapped
                if name.startswith("layers.") and ".engram.wkv" in name:
                    try:
                        h = header(wm[name])
                        ent = h.get(name)
                        if ent is not None:
                            eng += int(ent["data_offsets"][1] - ent["data_offsets"][0])
                    except Exception:
                        continue
            prof["engram_bytes"] = 2.0 * eng     # materialised as bf16
        except Exception:
            prof["engram_bytes"] = 2 * GB
        try:
            prof["expert_bytes"] = int(index.expert_blob_bytes())
        except Exception:
            pass
        # MTP / DSpark draft stages: three small decoders plus the markov head. Only
        # counted when the tier enables speculation, because they cost VRAM that the
        # resident layers would otherwise use.
        try:
            mtp = 0
            for name in wm:
                # routed experts inside the draft stages stream exactly like the main
                # model's (they are not resident): only attn/hc/markov are budgeted
                if name.startswith("mtp.") and ".ffn.experts." not in name:
                    try:
                        h = header(wm[name])
                        ent = h.get(name)
                        if ent is not None:
                            mtp += int(ent["data_offsets"][1] - ent["data_offsets"][0])
                    except Exception:
                        continue
            prof["mtp_bytes"] = 2.0 * mtp
        except Exception:
            prof["mtp_bytes"] = 0
    except Exception:
        pass
    return prof


def _spread_cards(cards: list[Card], n_layers: int, total_resident: int,
                  total_slots: int, ws_gb: float, head_gpu: bool,
                  layer_b: float = 0.19 * GB, max_layer_b: float = 0.72 * GB,
                  embed_b: float = 1.32 * GB, head_b: float = 1.32 * GB):
    """Distribute resident layers and expert slots across every CUDA device."""
    usable = {c.index: max(0.0, c.total_gb - 1.2) for c in cards}
    order = [cards[0].index] + [c.index for c in cards[1:]]
    caps: dict[int, int] = {}
    for idx in order:
        overhead = ws_gb + ((embed_b + (head_b if head_gpu else 0.0)) / GB if idx == order[0] else 0.0)
        caps[idx] = max(0, int((usable[idx] - overhead) / max(layer_b / GB, 1e-6)))
    total_cap = sum(caps.values()) or 1
    layer_card: dict[int, int] = {}
    assigned = 0
    for idx in order:
        want = min(caps[idx], max(0, int(round(total_resident * caps[idx] / total_cap))))
        for _ in range(want):
            if assigned >= n_layers:
                break
            layer_card[assigned] = idx
            assigned += 1
    # leftovers stream through the first card (its flat buffers are already there)
    while assigned < n_layers:
        layer_card[assigned] = order[0]
        assigned += 1
    assigned_per_card = {idx: 0 for idx in order}
    for _i, _c in layer_card.items():
        assigned_per_card[_c] = assigned_per_card.get(_c, 0) + 1
    spare = {}
    for idx in order:
        overhead = ws_gb + ((embed_b + (head_b if head_gpu else 0.0)) / GB if idx == order[0] else 0.0)
        spare[idx] = max(0.0, usable[idx] - overhead - assigned_per_card[idx] * (layer_b / GB))
    tot_spare = sum(spare.values()) or 1.0
    card_slots = {idx: max(0, int(total_slots * spare[idx] / tot_spare)) for idx in order}
    if total_slots and not any(card_slots.values()):
        card_slots[order[0]] = total_slots
    return layer_card, card_slots


def plan(hw: Hardware | None = None, n_layers: int = 40, model: dict | None = None,
         speculation: bool = False, calibrate: dict | None = None) -> Plan:
    """Solve the placement problem from bytes and bandwidths, never from tiers.

    Every number below is a measured price of the hardware (H2D bandwidth, the cost of
    a CPU head) or of the checkpoint (layer/expert/embedding bytes), so a machine that
    does not exist yet still gets a correct plan, and nothing degenerates into "if the
    card is 24GB then ...".
    """
    hw = hw or detect()
    model = model or {}
    # calibrate：启动期实测的标定结果（calibrate.calibrate 的返回）。
    # 规划器本身猜不出「这块盘的随机读在几个并发下最快」这类量，只能靠实测；
    # 传进来就用实测值，没传才退回推断。
    layer_b = float(model.get("layer_bytes", 0.19 * GB))
    max_layer_b = float(model.get("max_layer_bytes", 0.72 * GB))
    embed_b = float(model.get("embed_bytes", 1.32 * GB))
    head_b = float(model.get("head_bytes", 1.32 * GB))
    expert_b = float(model.get("expert_bytes", 18.8 * MB))
    n_experts = int(model.get("n_experts", 0) or 384)
    n_layers = int(model.get("n_layers", 0) or n_layers)

    V = hw.vram_budget_gb * GB
    cards = hw.cards or [Card(0, hw.name, hw.vram_gb, hw.vram_gb)]
    # No CUDA device at all: everything runs out of host memory (the engine has a CPU
    # path), so a "GPU head" would be a lie. Short-circuit the placement problem.
    has_gpu = bool(hw.cards) and V > 0

    H2D_BPS = 26.5e9            # measured pinned H2D
    PREFILL_TEMP = 0.60 * GB   # dequant/activation peak during prefill
    HEAD_CPU_COST_S = 0.053     # measured: 1.3GB bf16 GEMV on the host
    EMBED_CPU_COST_S = 0.001    # one 10KB row lookup
    # Measured on a live 8GB decode: 4.8GB = ring 2.16 + head 1.32 + 2x0.60 layers
    # + 0.12GB of workspace/KV. The workspace is small at decode; the big figure is
    # only needed transiently by a long prefill, and it is allocated then, not held.
    WS_BASE = 0.20 * GB         # fixed tilelang workspace
    # Measured from a live 8GB run: ring 2.16 + head 1.32 + one layer 0.60 + (workspace
    # + engram + KV) 1.22 = 5.3GB. The workspace scales with the wave but far less than
    # activations do: 0.02GB/token, not 0.15 (which over-reserved 1GB and cost 2 layers).
    WS_PER_TOKEN = 0.02 * GB
    LAYER_SLACK = 1.70          # allocator/state/fragmentation per resident layer,
                                # measured: 355MB of bf16 weights costs ~600MB of VRAM
                                # once its activations and fragments are counted
    # 显存里留给专家槽位的比例。0.70 是 8GB 卡上「槽位和 prefill wave 争显存」时的
    # 平衡点；在 44GB 的卡上，40 层全驻留之后剩下的显存几乎全可以用来装专家，
    # 而每多一个槽位就直接减少一次 H2D。实测 L20：1057 槽位时 8 个 token 做了
    # 770 次 pageable H2D（0.95ms/次 = 730ms，占整步的 61%），槽位一旦覆盖住
    # 热点专家，这部分直接归零。
    # 只留 10% 给 prefill wave / 运行时碎片。
    SLOT_FRACTION = 0.90
    # ring depth actually allocated by _flat_setup; TI_RING lets the two agree so the
    # planner does not reserve a buffer that is never allocated
    BUF_COUNT = max(2, min(4, int(__import__("os").environ.get("TI_RING", "3"))))

    # VRAM that is gone before a single layer is resident: the on-demand engram
    # projections, the KV window and the activation scratch. Ignoring these is what
    # produced plans that OOM'd in engram.forward / attention on an 8GB card.
    # engram projections are transient (only the engram layers hold them) and KV at
    # short context is small; 1.1GB was over-reserving by ~4 resident layers
    # Prefill runs the whole prompt at once: the fp8 dequant of wo_a (and of the
    # grouped wo matrix) allocates ~128MB per layer on top of the resident set, and
    # the prompt activations add more. Leaving the plan at zero margin put the peak
    # within 128MB of the card's limit, so the first large dequant OOM'd.
    # engram projections are DeepSeek-specific: a model without engram layers must not
    # reserve a gigabyte for them (Qwen reserved 1GB and streamed 5 of its 24 layers for
    # a mechanism it does not have)
    _engram_b = float(model.get("engram_bytes", 0.0) or 0.0)
    # 实测对账：8GB 卡上 plan 说 11 层 + 11 槽位占 4.15GB，实际初始化后占 6.21GB。
    # 差出来的主要是 MTP draft stages（mtp_bytes=1.33GB）。
    # 这里**不**无条件计入：第 421 行的 `if speculation:` 会在装不下时把它关掉，
    # 引擎于是根本不分配。两处必须用同一个判断，否则一边算进 used、一边不分配，
    # 槽位数白算少（实测无条件计入后 slots 从 11 掉到 1，而 MTP 明明是关的）。
    _mtp_b = float(model.get("mtp_bytes", 0.0) or 0.0)
    # 运行期临时量：forward 过程中才会发生的分配（dequant 展开的 fp32 中间量、
    # KV 增长、attention 临时张量、MoE workspace）。规划器必须为它留出空间，
    # 否则「能装下的槽位数」会大于「能算对的槽位数」。
    #
    # 0.15 是早期估算。它能满足「不崩」，但喂不饱「算得对」—— 这台 8GB 卡上
    # 槽位把 spare 吃到只剩几十 MB 时 forward 会静默算错。真实需求随卡型、
    # 上下文长度、并发数变化，所以这里只给默认值，由启动期标定（calibrate）
    # 实测后覆盖；换一台机器它自己会测出那个机器自己的值。
    _runtime_gb = 0.15
    if calibrate:
        _rv = calibrate.get('runtime_gb')
        if isinstance(_rv, (int, float)) and 0.0 < float(_rv) < 8.0:
            _runtime_gb = float(_rv)
    runtime_b = (_engram_b * 0.5 + _runtime_gb * GB + PREFILL_TEMP)
    if speculation and V < 16 * GB:
        # Drafting pays only when the acceptance rate is high; here it is 0.08 (docs/
        # analysis.md section 2), so the draft stages just cost resident layers and the
        # verify pass costs a full forward. Measured on the 8GB card: tier 2 with MTP
        # took 91s for 8 tokens against 50s for tier 1, same answer.
        speculation = False
    if speculation:
        # the draft stages and their windows are resident whenever speculation is on;
        # budget them instead of discovering them with an OOM after prefill
        runtime_b += float(model.get("mtp_bytes", 0.0)) + 0.35 * GB
        # ... but only keep them when the box can still hold the streaming ring plus a
        # couple of resident layers. On an 8GB card the draft stages cost more resident
        # layers than drafting ever repays, so the plan drops them and says so.
        # Drafting only pays if the box can still keep a useful resident set beside it:
        # each resident layer saves a layer-sized H2D per token, and a draft round only
        # adds value when accept > ~1 extra token. Four layers is the measured crossover
        # on this checkpoint (below it, MTP made a 3.0s step into a 3.7s one).
        # the streaming ring is sized by the largest layer in bf16, which measures 2x
        # the header-derived value (719MB vs 355MB on this checkpoint)
        _need = (BUF_COUNT * 2 * max_layer_b + WS_BASE + 8 * WS_PER_TOKEN
                 + float(model.get("engram_bytes", 0.6 * GB)) * 0.5 + 0.15 * GB
                 + float(model.get("mtp_bytes", 0.0)) + 0.35 * GB + 4 * layer_b * LAYER_SLACK)
        if _need > V:
            speculation = False
            runtime_b -= float(model.get("mtp_bytes", 0.0)) + 0.35 * GB

    # A tied checkpoint has no separate head: the head IS the embedding, so reserving
    # head_bytes a second time reserved 1.35GB that is never allocated.
    _tied_head = bool(model.get("tied_head", False))

    def layers_that_fit(embed_gpu, head_gpu, buffers, ws, slot_bytes=0.0):
        # a tied checkpoint has no separate head allocation (see _tied_head below)
        head_term = head_b if (head_gpu and not _tied_head) else 0.0
        # 专家槽位占的显存必须算进来。以前没算：规划器给出「11 层常驻 + 14 个槽位」，
        # 12 层吃 6333MB，14 个槽位再要 263MB，24GB 卡上剩下 333MB，第一次 prefill
        # 连 arena 都分配不出来（实测 !handles_ INTERNAL ASSERT），整个模型不可用。
        # 槽位数由 spare 推出，而 spare 又依赖这里 —— 用一次定点迭代解：先按 0 算，
        # 再按得到的槽位重算一次。收敛很快（槽位是 spare 的线性函数）。
        fixed = ((embed_b if embed_gpu else 0.0) + head_term
                 + buffers * max_layer_b + ws + runtime_b + float(slot_bytes))
        return int(max(0.0, V - fixed) // max(1.0, layer_b * LAYER_SLACK))

    ws0 = WS_BASE + 8 * WS_PER_TOKEN
    r_stream = layers_that_fit(True, True, BUF_COUNT, ws0)
    if r_stream >= n_layers:
        embed_gpu, head_gpu, buffers = True, True, 0
    else:
        best = None
        for eg, hg in ((True, True), (False, True), (False, False)):
            r = min(n_layers, layers_that_fit(eg, hg, BUF_COUNT, ws0))
            saved = max(0, r - min(r_stream, n_layers)) * layer_b / H2D_BPS
            price = (0.0 if eg else EMBED_CPU_COST_S) + (0.0 if hg else HEAD_CPU_COST_S)
            cost = price - saved
            if best is None or cost < best[0]:
                best = (cost, eg, hg, r)
        _, embed_gpu, head_gpu, _r = best
        buffers = BUF_COUNT
    # forced head placement (diagnostics/tuning: keeps 1.3GB of VRAM for resident layers
    # at the price of a host GEMV per token)
    if __import__("os").environ.get("TI_HEAD_CPU") == "1":
        head_gpu = False
    else:
        # _head_logits has no host-GEMV path (it slices the head on the device), so a
        # head_on_cpu plan only shows up later as a device mismatch. Keep the head
        # on-card unless the operator explicitly forces TI_HEAD_CPU.
        head_gpu = True
    # forced residency: overrides the solver so a caller can trade the streaming ring for
    # resident layers (the prefill path has to free them again, see Engine._forward_impl)
    _force_resident = __import__("os").environ.get("TI_RESIDENT")
    resident = min(n_layers, layers_that_fit(embed_gpu, head_gpu, buffers, ws0))
    # A model that fits entirely on the card has nothing to trade: pushing the embedding
    # off it would add a PCIe round trip per token and save memory that is not needed.
    # The cost model above assumes a streamed model, where one more resident layer is
    # worth a host lookup; that trade does not exist when nothing streams.
    if resident >= n_layers and has_gpu:
        if not embed_gpu:
            embed_gpu = True
            resident = min(n_layers, layers_that_fit(embed_gpu, head_gpu, buffers, ws0))
    if _force_resident is not None:
        # **TI_RESIDENT 是「就用这个数」，不是「至少这个数」。**
        #
        # 原来写的是 max(resident, ...)，只能往上加。但真正需要它的时候往往是
        # 往下减：resident 层占的是显存（每层 layer_b*1.7），而显存是槽位的
        # 唯一来源。少一层 resident 能多出十几个槽位，而 resident 层省下的只是
        # 一次 H2D（实测 1ms，已被异步掩盖），槽位省下的是每个专家 2.4ms 的
        # 装载 —— 10 倍以上的差价。要测这个方向，就得允许显式降低。
        try:
            resident = max(0, min(int(_force_resident), n_layers))
        except (TypeError, ValueError):
            pass
    if resident >= n_layers:
        buffers = 0
        resident = n_layers
    if not has_gpu:
        # everything runs from host memory on this box
        embed_gpu = head_gpu = False
        buffers = 0
        resident = 0
    elif resident == 0:
        # Too small for even one resident layer with the tables on the device: give the
        # VRAM back to the streaming ring and keep the expert slots minimal but non-zero
        # (a zero-slot plan allocates a fresh 18.8MB buffer per expert per layer per
        # token, which is the failure mode a 4GB card would hit).
        embed_gpu = head_gpu = False
        buffers = min(buffers, 2)
        resident = min(n_layers, layers_that_fit(False, False, buffers, ws0))

    # VRAM headroom knob: on cards where the plan uses ~85% of physical VRAM the driver
    # can page, which makes identical runs differ (docs/README.md 19). Lower residency
    # trades speed for a better chance of reproducibility.
    _cap = __import__("os").environ.get("TI_MAX_RESIDENT")
    if _cap:
        try:
            resident = max(0, min(resident, int(_cap)))
        except ValueError:
            pass

    # A handful of expert slots is worse than one fewer resident layer: without a
    # staging pool every expert allocates a fresh 18.8MB device buffer per layer per
    # token. Keep room for two layers' worth of experts (2 x top-k) before trading the
    # remaining VRAM for residency, which is a constraint, not a tier.
    # Slots are expert staging buffers. A model with no routed experts (Qwen, dense
    # models) must not reserve any: holding 12 x 18.8MB out of an 8GB card streamed 7 of
    # Qwen's 24 layers for nothing, and the model fits entirely.
    # **槽位是唯一路径，没有「无槽位」这个分支。**
    #
    # 原来的判据是 `expert_b > 0.5MB and topk > 0`，topk 从 model dict 里取，
    # 默认 0。于是任何没显式传 topk 的调用方都会得到 has_experts=False，
    # 进而 expert_b=0、slots=0，整个引擎退到「专家不进显存」的那条路 ——
    # 实测 5060 上就是这样：vram 6.9GB、只驻留 12 层，却算出 slots=0。
    #
    # 有路由专家就必须有槽位：模型几何里 expert_b 就是专家的真实字节数，
    # 它大于 0 就说明这个 checkpoint 有 routed experts，和 topk 传没传无关。
    # topk 只用来算「一步要几个槽位」（need_slots），不用来决定有没有槽位。
    _topk_hint = int(float(model.get("topk", 0) or 0))
    has_experts = expert_b > 0.5 * MB
    # 最少 1 个：再小的卡也要有一个能装专家的格子，计算时按需换进换出。
    # 12 是「两层 x topk 6」的下限，够 decode 一步的 MoE 不中途换出。
    MIN_SLOTS = max(1, min(12, 2 * _topk_hint)) if has_experts else 0
    if not has_experts:
        expert_b = 0.0

    # A tied checkpoint has no separate head: it IS the embedding, so counting both
    # reserved 1.35GB that is never allocated.
    _tied_head = bool(model.get("tied_head", False))

    def _used(r):
        return ((embed_b if embed_gpu else 0.0)
                + (head_b if (head_gpu and not _tied_head) else 0.0)
                + r * layer_b * LAYER_SLACK + buffers * max_layer_b + ws0 + runtime_b)

    used = _used(resident)
    spare = max(0.0, V - used)
    if not _force_resident:
        # the solver only keeps slot room when it chose the residency itself; a caller
        # that asked for a specific count (TI_RESIDENT) wants those layers, not slots
        while resident > 0 and int(spare * SLOT_FRACTION // max(1.0, expert_b)) < MIN_SLOTS:
            resident -= 1
            used = _used(resident)
            spare = max(0.0, V - used)
    if not has_experts:
        slots = 0
    else:
        slots = int(spare * SLOT_FRACTION // max(1.0, expert_b))
        slots = max(slots, 2) if has_gpu else 0

    # ---- expert coverage beats a device-resident table ---------------------------
    # A token can route to topk*n_layers distinct experts. Once the slot pool covers
    # all of them the expert IO term (topk*n_layers*18.8MB per token = 4.5GB, ~1s at
    # 4.5GB/s) disappears; the price of moving the 1.32GB embedding/head table to the
    # host is one 10KB row lookup per token (~1ms). On a 24GB card that is 5 -> 80 tok/s.
    need_slots = int(model.get("topk") or 0) * n_layers

    def _used_with(r, eg, hg):
        return ((embed_b if eg else 0.0) + (head_b if hg else 0.0)
                + r * layer_b * LAYER_SLACK + buffers * max_layer_b + ws0 + runtime_b)

    if need_slots and has_gpu and slots < need_slots * 0.15:
        # Below ~15% coverage the VRAM slot pool used to cost more than it saved: every
        # non-resident expert paid a DMA plus an event wait, and a slot marked "ready"
        # from an earlier use could hand back the wrong bytes (96 slots produced a
        # different first token; 13 made two identical greedy runs disagree).
        # Those two defects are fixed -- the pool writes its own slot table and never
        # hands back bytes it did not just place (tests/test_slot_pool.py) -- so the
        # reason to zero the pool is gone. Keep the slots: on a small card they are the
        # only thing standing between decode and a disk read per expert.
        pass
    if need_slots and has_gpu and slots < need_slots and (embed_gpu or head_gpu):
        # **判据是「槽位显著变多」，不是「槽位能覆盖整步」。**
        #
        # 原来只有 _slots2 >= need_slots（240 个）才切换。240 在 8GB 卡上
        # 根本达不到（embed 挪走也只到 91），所以这条分支从来没生效过 ——
        # 规划器永远给 embed 留在显存、槽位只有 24 个的结果。
        #
        # 代价对得上：embed 留在显存省的是每 token 一次 10KB 行查找（约 1ms），
        # 而多出来的 67 个槽位（24 -> 91）能省掉每 token 67 个专家的搬运
        # （约 1.2GB，盘上 0.3s）。1ms 换 300ms，任何比例下都该换。
        #
        # 阈值取 1.25 倍：切换本身没有代价（embed 走主机查表），但要有明显收益
        # 才值得改变计划，避免在边界上来回横跳。
        _best = None
        for _eg, _hg in ((False, False), (False, True), (True, False)):
            _r2 = min(n_layers, layers_that_fit(_eg, _hg, buffers, ws0))
            _slots2 = int(max(0.0, V - _used_with(_r2, _eg, _hg)) * SLOT_FRACTION
                          // max(1.0, expert_b))
            if _best is None or _slots2 > _best[0]:
                _best = (_slots2, _eg, _hg, _r2)
            if _slots2 >= need_slots:
                _best = (_slots2, _eg, _hg, _r2)
                break
        if _best is not None and _best[0] >= max(slots * 1.25, slots + 8):
            slots, embed_gpu, head_gpu, resident = _best[0], _best[1], _best[2], _best[3]
    # ---- 定点迭代：槽位自己也要占显存 -------------------------------------
    # 上面的 resident 和 slots 是分别算的，两者加起来会超。实测 40 层模型
    # （层权重 8GB 卡上 164MB/层）：plan 给 resident=11、slots=14，初始化后
    # 只剩 333MB，第一次 prefill 的 arena 直接分配失败
    # （!handles_.at(i) INTERNAL ASSERT），整个模型不可用。
    # 这里迭代：把 slots*expert_b 记入 used，重算 resident 和 slots。
    if has_experts and has_gpu and slots > 0:
        for _ in range(6):
            _sb = slots * expert_b
            # 注意：`_used_with` 与 `layers_that_fit` 的记账口径必须一致，且**只扣一次**。
            # 之前这里既在 layers_that_fit 里传 slot_bytes、又在外面减 _sb，
            # 槽位被算了两次，24GB 卡从 268 掉到 208（不够覆盖 240 个可路由专家，
            # 直接退化成每 token 有磁盘 IO）。
            # `layers_that_fit` 内部已经把 slot_bytes 记进 fixed 了，这里再减一次
            # 就是重复扣减 —— 注释里说 24GB 卡从 268 掉到 208 那次只修了外面，
            # 里面这一处仍在重复：实测 8GB 卡上表算给出 77 个槽位，最终只有 8 个。
            _r = min(n_layers, layers_that_fit(embed_gpu, head_gpu, buffers, ws0,
                                              slot_bytes=_sb))
            _spare = max(0.0, V - _used_with(_r, embed_gpu, head_gpu))
            _s = int(_spare * SLOT_FRACTION // max(1.0, expert_b))
            if _r == resident and _s == slots:
                break
            resident, slots = _r, max(0, _s)
        # 兜底：大卡上装得下全模型的层时，槽位不该比「单会话可路由的专家数」还少
        # ——那是最基本的覆盖率，低于它每次 token 必然有磁盘 IO。
        if resident >= n_layers and need_slots:
            slots = max(slots, min(need_slots,
                                   int(max(0.0, V - _used_with(resident, embed_gpu,
                                                                  head_gpu))
                                       // max(1.0, expert_b))))

    # The wave size is whatever the spare VRAM can hold at WS_PER_TOKEN each; the lower
    # bound keeps a chunk from degenerating into single rows. The old upper bound of 64
    # came from a measurement of the tilelang kernel set ("a wave of 8 needs ~1.6GB and
    # 165s for a 21-token prompt on 8GB; 16 needs 63s but pages") -- that code is gone
    # and the bound was capping every large card at 64 tokens per wave, which is also
    # below the M that makes the tensor-core prefill path worthwhile. Leave the ceiling
    # high enough that the spare-VRAM formula decides: on this 8 GB box it still lands
    # on 8, and a 141 GB card gets waves in the hundreds.
    chunk = int(max(0.0, spare * (1.0 - SLOT_FRACTION)) // max(1.0, WS_PER_TOKEN))
    chunk = max(8, min(4096, chunk))

    # ---- host side: leave the page cache and the CPU tables their memory -----------
    # The expert cache is not free: every GB it takes is a GB the OS cannot use to keep
    # shard data warm, and the measured effect is *negative* past a point (cache 11GB ->
    # 1.45s fetch, cache 15.3GB -> 2.65s fetch on the same 47GB box). Size it from what
    # is genuinely left after the slab, the tables kept on the host, and ~6GB of page
    # cache, then take half of that as the cache.
    host = hw.ram_budget_gb * GB
    if hw.unified:
        # same physical pool: subtract what the device side already booked
        host = max(0.0, host - V)
    # The pinned slab is a PREFETCH buffer, not one slot per streamed layer: the ring is
    # BUF_COUNT deep. Booking (n_layers - resident) * layer_b assumed every streamed layer
    # needs its own pinned copy, which for a dense 27B (63 layers x 1441MB) claimed 88GB
    # of a 47GB box and collapsed the host cache to its 256MB floor -- the model then read
    # every layer from disk on every token.
    slab_b = min(max(0.0, (n_layers - resident) * layer_b),
                 max(1, buffers) * max_layer_b)
    cpu_tables = (embed_b if not embed_gpu else 0.0) + (head_b if not head_gpu else 0.0)
    page_cache_b = min(6.0 * GB, 0.20 * host)
    # 专家缓存的大小：扣掉层 slab、CPU 侧 embed/head、页缓存之后，剩下的给它。
    #
    # 以前乘 0.8，理由是「47GB 机器上 4.8GB 太小、10GB 足够」。搬到 371GB 的机器上，
    # 那个 0.8 意味着 50GB 内存白留 —— 而模型**全部 15360 个专家只要 268.9GB**。
    # 实测 L20：pool 只拿到 231.4GB = 13214 项，比需要的少 2146 项（14%）。
    # 于是预热装不下全部，decode 每步有 13-25 个专家要去读盘，
    # 每层一次 `_miss_scan` 全同步，GPU 利用率被压到 0-16%。
    #
    # 改法：**只要装得下就装全部**。判据用「全部专家字节 <= 剩余预算的 92%」，
    # 那个 8% 留给预热过程中的临时 buffer 和页缓存抖动。装不下时才退回 0.8。
    _room = max(0.0, host - slab_b - cpu_tables - page_cache_b)
    _all_experts_b = float(n_layers) * float(n_experts) * float(expert_b)
    if _all_experts_b > 0 and _all_experts_b <= _room * 0.92:
        # 装得下全部：给足，让 decode 稳态零读盘
        cache_b = int(max(256 * MB, _all_experts_b * 1.02))
    else:
        cache_b = int(max(256 * MB, _room * 0.8))

    if len(cards) <= 1:
        layer_card = {i: 0 for i in range(n_layers)}
        card_slots = {0: slots}
    else:
        layer_card, card_slots = _spread_cards(cards, n_layers, resident, slots,
                                               ws0 / GB, head_gpu, layer_b,
                                               max_layer_b, embed_b, head_b)
    fixed_b = ((embed_b if embed_gpu else 0.0) + (head_b if head_gpu else 0.0)
               + buffers * max_layer_b + ws0 + runtime_b)
    fits = fixed_b <= V
    if not fits:
        need_gb = (fixed_b + 1.5 * max(1, len(cards) or 1) * GB) / GB
        reason_extra = f" INSUFFICIENT VRAM: needs about {need_gb:.1f}GB for the streaming skeleton"
    else:
        reason_extra = ""
    reason = (
        f"{hw.name} vram={hw.vram_gb:.1f}GB({len(cards)} card) ram={hw.ram_gb:.1f}GB"
        f"(avail {hw.ram_avail_gb:.1f}) -> layers={resident}/{n_layers}"
        f" stream={n_layers - resident} buf={buffers} slots={slots}"
        f" cache={cache_b/GB:.1f}GB chunk={chunk}"
        f" embed={'gpu' if embed_gpu else 'cpu'} head={'gpu' if head_gpu else 'cpu'}"
        f" layer={layer_b/MB:.0f}MB expert={expert_b/MB:.1f}MB{reason_extra}"
    )
    # ---- 主机侧账面：这里算完，别处直接用 ------------------------------
    #
    # slab 源：给每个 streamed 层留一份 pinned 源。每层按 max(layer_bytes,
    # max_layer_bytes/2) 估 —— 实测值就是这个量级（header 的 layer_bytes 是
    # tensor 字节均值，不含对齐和额外 buffer；max/2 才接近实际 span）。
    # 1.15 是构建期临时量（一层在 host 上读出来、拷进 slab、再释放旧的）。
    # 每层 slab 源的实际字节：header 的 layer_bytes 是各 tensor 字节均值，不含
    # 4096 对齐和额外 buffer；实测 span 是它的 1.4 倍左右，而 max_layer_bytes 的
    # 一半正好落在实测值附近（本机 172MB*(1.4) vs 468MB/2=234MB vs 实测 237MB）。
    # 取两者较大：既不会因为某层特别胖而少留，也不会因为 header 均值偏小而少留。
    _slab_per_layer = max(layer_b * 1.4, max_layer_b * 0.5)
    _n_streamed = max(0, n_layers - resident)
    # 每个 streamed 层一份源，再加「一层正在构建」的临时窗口（读进来 -> 拷进源
    # -> 释放旧的，同一时刻只有一层在途）。这个窗口就是 _slab_per_layer 本身，
    # 所以不需要再额外塞一个跟机器无关的常数。
    slab_reserve = int((_n_streamed + 1) * _slab_per_layer) if _n_streamed else 0
    # 宿主池：物理可用量减掉进程自身的常驻部分。**上限只取物理可用**，
    # 绝不把 pagefile 算进来（那是硬盘冒充内存，会让 pinned 页落到 swap）。
    _ram_avail = float(getattr(hw, "ram_avail_gb", 0) or 0) * GB
    _ram_total = float(getattr(hw, "ram_gb", 0) or 0) * GB
    if _ram_avail > 0:
        # **全给池子，不减余量。** 加载期临时、torch 自身那些开销由 claim()
        # 的失败路径兜底（拿不到就缩）。预设 15% 在 47GB 上是 7GB、在 371GB
        # 上是 55GB，两个数都没有依据，而且会让「这台机器能用多少」变成
        # 一个写死的猜测。
        host_pool = int(_ram_avail)
    elif _ram_total > 0:
        host_pool = int(max(0.0, _ram_total * 0.5))
    else:
        host_pool = 0
    # 池子必须装得下「slab + 缓存」，否则一开始就注定要打架
    _need = int(cache_b) + int(slab_reserve)
    if host_pool and _need > host_pool:
        # 缓存让路：slab 是常驻的（省的是每步的层重读），缓存是机会性的
        cache_b = max(192 * MB, min(int(cache_b), host_pool - int(slab_reserve) - 192 * MB))
    # pin 预算 = 缓存的全部。pageable 的专家要多走一次 staging memcpy 才能上
    # 显存，没有任何理由只 pin 一半（以前写死 0.5，实测 240 个专家每步有一半
    # 走 pageable，单步多花 1 秒）。
    pin_budget = int(cache_b)
    # 读并发：SSD 队列深度，与机器大小无关
    # pin 比例：训练器扫出来的值优先。没有就先用 0.5（训练器能给的最保守起点）。
    _pin_ratio = 0.5
    if isinstance(calibrate, dict):
        _pr = calibrate.get("pin_ratio")
        if isinstance(_pr, (int, float)) and 0.0 < float(_pr) <= 1.0:
            _pin_ratio = float(_pr)
    # 并发深度：环境变量 > 实测标定 > 推断。
    #
    # 这个数没有「通用正确值」：SSD 的队列深度饱和点取决于盘、驱动和文件在盘上
    # 的物理分布。本机实测 {4:2.8, 8:3.0, 16:3.3, 32:3.1}GB/s，最优在 16；
    # 另一个时刻同一台机器又测出 8 最优 —— 所以它是「这块盘此刻的最优」，
    # 不是常数。calibrate 每次启动实测一遍并缓存。
    _rt = __import__("os").environ.get("TI_READ_THREADS")
    read_threads = 0
    if _rt:
        try:
            read_threads = max(1, min(64, int(_rt)))
        except (TypeError, ValueError):
            read_threads = 0
    if not read_threads and isinstance(calibrate, dict):
        _v = calibrate.get("read_threads")
        if isinstance(_v, int) and _v > 0:
            read_threads = max(1, min(64, _v))
    if not read_threads:
        # 没有实测值时的保守推断：SSD 队列深度经验区间 8..32，取中值。
        read_threads = 16

    return Plan(
        layer_card=layer_card,
        card_slots=card_slots,
        speculation=bool(speculation),
        embed_on_cpu=not embed_gpu,
        gpu_layers=resident,
        head_on_cpu=not head_gpu,
        expert_slots=slots,
        cache_bytes=cache_b,
        prefill_chunk=chunk,
        stream_layers=n_layers - resident,
        reason=reason,
        fits=fits,
        pin_ratio=float(_pin_ratio),
        host_pool_bytes=int(host_pool),
        read_threads=int(read_threads),
    )
