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
    # ---- 带宽：规划器做取舍唯一该看的「时间维度」 ----------------------------
    #
    # 没有这几个数，规划器就只能用模块常数（H2D_BPS=26.5e9 是这台 5060 上测的），
    # 于是 GB300 和 2080Ti 共用同一个带宽做决策 —— 那个「时间判据」在别的卡上
    # 是错的，而且会系统性偏向某一侧（慢卡上低估 H2D 代价，于是过多驻留层）。
    #
    # 单位 byte/s。None 表示未知，由调用方从卡型/PCIe 代次推或让训练器实测。
    h2d_bps: float | None = None      # 主机 pinned -> 显存
    disk_bps: float | None = None     # 顺序读（checkpoint 冷读）
    ram_bps: float | None = None      # 内存 -> 显存（命中缓存时走这条）
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

    def bandwidth(self, calibrate: dict | None = None,
                  model_dir: str | None = None) -> dict:
        """这台机器的三条带宽，byte/s。**现场测，没有默认值。**

        规划器做取舍全靠这三个数：常驻一层省下来的 H2D、一次磁盘读、一次
        内存命中的代价。它们必须是这台机器的，不能是模块常数（那会让 GB300
        和 2080Ti 用同一个带宽做决策）。

        三项都能测出来：

          h2d_bps   把一块 pinned 内存搬进显存并计时
          disk_bps  顺序读一块数据并计时
          ram_bps   memcpy 一大块并计时

        结果按 (机器指纹, 模型目录) 缓存在 tuning.json 里，第二次启动不再测。
        """
        cached = dict(calibrate or {})
        if all(cached.get(k) for k in ("h2d_bps", "disk_bps", "ram_bps")):
            return {k: float(cached[k]) for k in ("h2d_bps", "disk_bps", "ram_bps")}

        out = {}
        # --- H2D：必须用引擎真实的搬运方式，不能是裸 copy_
        #
        # 裸 copy_ 测出来 7.1GB/s，但那条路 torch 会先做 pageable staging memcpy
        # （见 loading._async_h2d 的注释）。引擎实际走 cudaMemcpyAsync 直发，
        # 两者不是一回事 —— 测错的话规划器会按「很慢」布局，过度驻留。
        #
        # 所以这里直接调 _async_h2d（和装载路径同一个函数），块大小取真实层的
        # 量级（174MB），并且先跑两轮预热（第一次会建 context/触碰页）。
        out["h2d_bps"] = 1.0
        try:
            import time as _t

            import torch

            from .loading import _async_h2d

            n = 174 << 20
            cpu = torch.empty(n, dtype=torch.uint8, pin_memory=True)
            gpu = torch.empty(n, dtype=torch.uint8, device="cuda")
            ok = 0
            for _ in range(2):
                if _async_h2d(gpu, cpu, n):
                    ok += 1
                torch.cuda.synchronize()
            if ok:
                t0 = _t.perf_counter()
                for _ in range(3):
                    _async_h2d(gpu, cpu, n)
                torch.cuda.synchronize()
                dt = (_t.perf_counter() - t0) / 3
                out["h2d_bps"] = float(n / max(dt, 1e-9))
        except Exception:
            pass

        # --- disk：必须读真实 checkpoint 分片，不能读临时文件
        #
        # 临时文件测出来 8.7GB/s —— 那是页缓存直接命中（刚写进去的数据还在内存里），
        # 不是盘的速度。而专家 blob 的真实读取模式是「从几十 GB 的分片里随机 seek」，
        # 两者能差一个数量级。
        #
        # calibrate.measure_disk 已经在做正确的事（读真实分片、区分 seq/rand、用
        # readinto 绕开对象分配），直接复用。
        out["disk_bps"] = 1.0
        out["disk_rand_bps"] = 1.0
        try:
            from .calibrate import measure_disk

            md = measure_disk(model_dir) if model_dir else {}
            if md.get("seq_gbps"):
                out["disk_bps"] = float(md["seq_gbps"]) * 1e9
            if md.get("rand_gbps"):
                out["disk_rand_bps"] = float(md["rand_gbps"]) * 1e9
        except Exception:
            pass

        # --- ram：memcpy
        try:
            import ctypes

            n = 128 << 20
            a = ctypes.create_string_buffer(n)
            b = ctypes.create_string_buffer(n)
            t0 = __import__("time").perf_counter()
            for _ in range(3):
                ctypes.memmove(b, a, n)
            dt = (__import__("time").perf_counter() - t0) / 3
            # memmove 一次读写各 n 字节，有效带宽是 2n/dt
            out["ram_bps"] = float(2 * n / max(dt, 1e-9))
        except Exception:
            out["ram_bps"] = 1.0

        return out


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


