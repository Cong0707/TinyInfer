"""tinyinfer engine: full-model driver on top of adapters + tiers + ExpertStore.

Owns model lifecycle: embedding, block residency (lazy per-layer load), generation.
The fetch/dequant/eviction policy lives in loading.py; this is the orchestration layer.
"""
from __future__ import annotations

# cuBLAS GEMMs are not bit-reproducible by default (split-k order varies run to run),
# and this engine samples greedily: a one-ulp logit difference sends the first token
# down a different path (observed: "Paris" in one run, an unrelated essay in the next).
# PyTorch's prescribed fix is this workspace config, and it must be set before CUDA
# initialises -- hence module import time.
import os as _os_cublas

_os_cublas.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from pathlib import Path

from .adapters import ModelAdapter, open_adapter
from .config import TinyConfig
from .deepseek.hc import hc_split_sinkhorn  # noqa: F401  (re-export for callers)
from .nn import rms_norm, sample_logits, softmax_sample
from .deepseek.model import Block, set_weight
from .tiers import OptLevel, TierConfig, plan

import torch
import torch.nn.functional as F


class Engine:
    @classmethod
    def from_hf(cls, source: str | Path, **kwargs) -> "Engine":
        # TI_TIER lets a caller pick the optimisation level without touching code
        import os as _os

        if "tier" not in kwargs and _os.environ.get("TI_TIER"):
            kwargs["tier"] = OptLevel(int(_os.environ["TI_TIER"]))
        """Load directly from a HF repo id / hub snapshot / flat checkpoint dir.

        This is the canonical entry point: no per-deployment model-dir plumbing.
        Local-only by default; pass local_files_only=False to plan.resolve_hf_snapshot
        (via open_adapter) once network downloads are acceptable.
        """
        return cls(open_adapter(source), **kwargs)

    def close(self) -> None:
        """释放这个引擎占住的一切，之后可以再建一个。

        **原来没有这个方法，这是服务的实际泄漏点。** 引擎会 claim 一块几十 GB 的
        主机池（cudaHostAlloc / VirtualAlloc）、占满显存槽位、持有 engine lock；
        这些资源在进程退出前都不会自己消失。于是「换模型」「重载配置」这类操作
        每做一次就多留一份池子 —— 表面上看是内存只涨不跌，实际是旧引擎的东西
        没人收。

        释放顺序是按「谁依赖谁」倒着来，不是随便排的：
          1. CUDA graph：它持有捕获期的中间 buffer，先拆掉才不会在释放权重时
             被它的引用挡住；
          2. 显存槽位池：释放 device 侧的大块，把显存还给驱动；
          3. 主机池：cudaFreeHost / VirtualFree，这是真正的大头；
          4. engine lock：最后放，保证上面几步期间没有第二个引擎进来。

        任何一步失败都不中断后面的步骤 —— 关不干净比关得慢更糟，但一步失败就
        全放弃会让剩下的资源彻底留在那里。
        """
        import contextlib
        import gc

        # 0) **先停后台预热线程。** 它是个 daemon，一直在往主存池和专家缓存里写；
        #    池子被拆掉之后它还在写，就是 0xC0000005（实测 bench 跑完后 close 必崩）。
        #    停线程必须排在任何释放动作之前。
        with contextlib.suppress(Exception):
            _store0 = getattr(self, "expert_store", None)
            _mgr0 = getattr(_store0, "_tier_mgr", None)
            if _mgr0 is not None and hasattr(_mgr0, "stop_warm"):
                _mgr0.stop_warm()

        # 1) 再把所有指向池子的 Python 引用断干净。
        #
        # **这一步不能省，否则会直接崩在访问违例。** 主机池里切出去的内存被
        # torch.frombuffer 包成了 tensor，层权重、blob pool、KV 都持有它们。
        # 如果先 cudaFreeHost 再清引用，那些 tensor 就成了野指针，析构或读一下
        # 就是 0xC0000005（实测 bench 完整跑完后在 close 处崩掉）。
        # 所以顺序是「断引用 -> 强制回收 -> 确认没有残留 -> 才还内存」。
        with contextlib.suppress(Exception):
            g = getattr(self, "_graph", None)
            if isinstance(g, dict):
                for _v in list(g.values()):
                    if hasattr(_v, "reset"):
                        with contextlib.suppress(Exception):
                            _v.reset()
            self._graph = None

        # 1) 层权重和 blob pool：真正持有池子切片的那些对象
        with contextlib.suppress(Exception):
            self._blocks = {}
        with contextlib.suppress(Exception):
            _store = getattr(self, "expert_store", None)
            _slots = getattr(_store, "slots", None)
            if _slots is not None:
                for _attr in ("_buffers", "_arena", "_staging", "_by_key"):
                    with contextlib.suppress(Exception):
                        setattr(_slots, _attr, None if _attr != "_by_key" else {})
                with contextlib.suppress(Exception):
                    _slots._n = 0
            if _store is not None:
                with contextlib.suppress(Exception):
                    _store._pool = None
        with contextlib.suppress(Exception):
            self.host_pool = None
        with contextlib.suppress(Exception):
            self.expert_store = None

        # 2) 强制回收，让上面断掉引用的 tensor 真的析构
        with contextlib.suppress(Exception):
            for _ in range(3):
                gc.collect()

        # 3) 显存还回去
        with contextlib.suppress(Exception):
            import torch as _torch

            if _torch.cuda.is_available():
                _torch.cuda.empty_cache()

        # 4) 主机池：这是几十 GB 的那一块，必须最后动
        with contextlib.suppress(Exception):
            gc.collect()
            from .budget import budget as _budget_fn

            _b = _budget_fn()
            _freed = int(_b.close() or 0)
            if _freed:
                print(f"[tinyinfer] released {_freed / (1 << 30):.1f}GB host pool",
                      flush=True)

        # 5) engine lock 最后放
        with contextlib.suppress(Exception):
            self._engine_lock.release()
            self._engine_lock = None

    def __enter__(self) -> "Engine":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __init__(
        self,
        adapter: ModelAdapter | str,
        *,
        tier: OptLevel | TierConfig = OptLevel.AGGRESSIVE,
        expert_store=None,
        max_layers: int | None = None,
        allow_missing_layers: bool = False,
        device: str = "cpu",
        tokenizer=None,
        gpu_layers: int | None = None,
        layers: list[int] | None = None,
        head_on_cpu: bool = False,
    ):
        # ONE engine per machine: the engine claims a large pinned host pool and the
        # whole GPU, so a second instance doubles the RAM (the box swaps) and on Windows
        # makes later cudaMalloc calls fail. Refuse before reading or allocating anything
        # rather than degrade the engine that is already running.
        from .runlock import EngineLock

        self._engine_lock = EngineLock().acquire()
        self.layer_ids = list(layers) if layers is not None else None
        self.device = device
        # Every visible card is probed and pooled, unconditionally. There is no separate
        # single-GPU code path: one card is simply the case where the device list has one
        # entry, so placement, addressing and the expert slot split all run the same code
        # and produce a one-element plan. That matters beyond tidiness -- a special case
        # for N=1 is a path nobody exercises at N>1 and vice versa, and the two drift.
        #
        # The pool accounts for what is *really* free on each card, so the split is
        # asymmetric without a knob: the caller never says "put 60% here", the cards say
        # how much they can hold.
        self.devices: list = []
        self.pool = None
        self.placement = None
        if str(device).startswith("cuda"):
            try:
                from .tp import Pool, Placement, probe_devices

                self.devices = probe_devices()
                if self.devices:
                    self.pool = Pool(self.devices)
                    self.placement = Placement(self.pool)
                    desc = ", ".join(
                        f"cuda:{d.index} {d.name} {d.free >> 30}GiB free "
                        f"({d.bandwidth / 1e9:.0f} GB/s)" for d in self.devices)
                    print(f"[tinyinfer] GPU pool ({len(self.devices)}): {desc}", flush=True)
            except Exception as _tp_exc:
                print(f"[tinyinfer] GPU probe skipped ({_tp_exc!r})", flush=True)
        self._layer_dev: dict[int, int] = {}
        self._expert_dev: dict[tuple[int, int], int] = {}
        # Padding token for batched prefill. Right padding relies on the causal mask, so
        # any id works numerically; EOS is the conventional one and is what the tokenizer
        # pads with anyway.
        self.pad_id = 0
        self._tokenizer = tokenizer
        if tokenizer is not None:
            _p = getattr(tokenizer, "pad_token_id", None)
            if _p is None:
                _p = getattr(tokenizer, "eos_token_id", None)
            if _p is not None:
                self.pad_id = int(_p)
        self.hashes = None
        self.adapter = open_adapter(adapter) if isinstance(adapter, str) else adapter
        # the architecture family drives the two branches the engine cannot abstract away
        # (how a layer is built, and how its KV is stored)
        try:
            self.family = self.adapter.load_config().arch_family()
        except Exception:
            self.family = getattr(self.adapter, "family", "")
        self.cfg = self.adapter.load_config()
        self.store = self.adapter.store
        # tier first: the hardware plan reserves MTP when the tier enables speculation
        if isinstance(tier, str):
            from .tiers import plan as _tier_plan
            tier = _tier_plan(OptLevel[tier.upper()])
        self.tier = plan(tier) if isinstance(tier, OptLevel) else tier
        if __import__("os").environ.get("TI_DETERMINISTIC") == "1":
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.deterministic = True
        if __import__("os").environ.get("TI_TORCH_ONLY") == "1":
            # diagnostic: pure-torch numerics, no tilelang/fp4 kernels
            self.tier.use_tilelang_attn = False
            self.tier.use_fp4_gemm = False
            self.tier.use_fp8_gemm = False
        _topk = int(__import__("os").environ.get("TI_TOPK", "0"))
        if _topk > 0:
            self.tier.expert_topk = _topk
        # kv_bits was a config field nobody read: KV precision rode on the weight flag
        # instead, so tier 2's kv_bits=8 had no effect and a full-precision KV cache was
        # silently the only option. Derive the flag here, once, and hand it to every
        # Block. 16 = full precision (no round trip), 8 = the fp8 window cache the
        # reference uses. 4 has no implementation -- say so instead of pretending.
        # approximations: the dict was carried around but never consumed, so tier 3's
        # "every8 skip / low-rank wo / fast math" were documentation only. Wire the two
        # that change which weights are read (skip) and how the FFN output is projected.
        _approx = dict(getattr(self.tier, "approximations", {}) or {})
        self._approx = _approx
        self._skip_layers: set[int] = set()
        _pat = _approx.get("skip_layer_pattern")
        if _pat:
            from .approx import skip_layer_plan as _skip_plan
            _every = {"every8": 8, "every4": 4, "every2": 2,
                      "every16": 16}.get(str(_pat))
            if _every:
                self._skip_layers = set(_skip_plan(self.cfg.n_layers, _every))
            else:
                raise ValueError(f"unknown skip_layer_pattern {_pat!r}; known: every2/4/8/16")
        self._wo_low_rank = float(_approx.get("wo_low_rank") or 0.0)
        self._fast_math = set(_approx.get("fast_math") or ())
        if self._fast_math:
            # 'exp','silu','rsqrt','sin','cos' -> torch's fast-math kernels. There is no
            # torch-level switch for one op at a time, so this turns on the global
            # approximate-math flags those kernels honour on this backend.
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            except Exception:
                pass
        _kvb = int(getattr(self.tier, "kv_bits", 16) or 16)
        self._kv_bits_effective = 16 if _kvb >= 16 else 8
        # The KV cache used to take its precision from keep_weights_quantized (one flag
        # drove both axes), and the hardware plan was tuned against that. kv_bits now
        # drives it, but only as an override: a tier that leaves kv_bits at 16 keeps the
        # weight flag's behaviour so the resident-layer plan does not change.
        _kvb_explicit = bool(getattr(self.tier, "kv_bits", 0))
        if not _kvb_explicit or _kvb >= 16:
            self._kv_quant = bool(self.tier.keep_weights_quantized)
        elif _kvb == 8:
            self._kv_quant = True
        else:
            # 4-bit KV has no reader in this engine. Tier 3 is the experimental tier and
            # has to launch, so fall back to the narrowest precision that does exist
            # (fp8) and say so -- silently pretending kv4 was on is what made the field
            # useless in the first place.
            self._kv_quant = True
            self._kv_bits_effective = 8
            print(f"[tinyinfer] kv_bits={_kvb} requested by tier "
                  f"{self.tier.level.name}: no 4-bit KV reader exists, using fp8 "
                  f"(kv_bits=8) instead", flush=True)
        # The streaming ring, the resident layers and the attention temporaries are
        # allocated and freed at very different lifetimes; on a card this close to full
        # the allocator's default segmenting leaves 0 free bytes even when the total is
        # under capacity. expandable segments lets PyTorch unmap rather than fragment.
        import os as _os2
        _os2.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        # hardware-derived plan (VRAM/RAM budget -> resident layers, slots, cache, chunk)
        from .resources import model_profile as _model_profile
        from .resources import plan as _hw_plan
        if device != "cpu":
            # measure the checkpoint: per-layer / expert / embed / head bytes come from
            # the safetensors headers, so the solver multiplies real numbers instead of
            # a table tuned for one model on one card
            try:
                from .loading import ShardIndex

                _prof = _model_profile(ShardIndex(self.store.model_dir))
                self._model_prof = _prof
                if _prof is not None:
                    # routing width decides whether the expert pool can cover the experts
                    # a token may route to (planner's coverage/slot decision)
                    _prof["topk"] = int(getattr(self.cfg, "n_activated_experts", 0) or 0) or 0
                    # MTP 的 draft stages 真的会被构造吗？规划器不能靠猜：它要么占
                    # 显存要么不占，而「会不会加载」由 tier.mtp 决定。实测 8GB 卡上
                    # 规划器算 11 层占 4.15GB、实际占 6.21GB，差的就是这 1.33GB。
                    _prof["mtp_resident"] = bool(getattr(self.tier, "mtp", False))
                    # a tied checkpoint's head IS its embedding: reserving head_bytes
                    # again costs the plan 1.35GB of residency it never allocates
                    # 只有真的「没有独立 head」或配置声明 tied 才算 tied。
                    # 这个 checkpoint 的 head_key 存在、tie_word_embeddings 为 False，
                    # 而实测 embed 和 head 两张表前 4KB 字节就不同 —— 它们是两张
                    # 独立的 1.23GB 权重。规划器以前按 tied 处理，只算了一份，
                    # 于是 8GB 卡上 resident 多给了 5 层，40 层模型连槽位 arena
                    # 都分配不出来（!handles_ INTERNAL ASSERT）。
                    _prof["tied_head"] = (getattr(self.adapter, "head_key", "x") is None
                                          or bool(getattr(self.cfg, "tie_word_embeddings",
                                                          False)))
            except Exception:
                _prof = None
            # speculation reserves its draft stages up front (tier 2 is the default)
            self.hw_plan = _hw_plan(None, self.cfg.n_layers, model=_prof,
                                    speculation=bool(getattr(self.tier, "mtp", False)))
        else:
            self.hw_plan = None
        if self.hw_plan is not None:
            print(f"[tinyinfer] {self.hw_plan.reason}", flush=True)
            # a model that keeps materialised layers needs its ceiling now, not on the
            # first block() call: the plan already knows the host budget
            if getattr(self.adapter, "family", "") == "qwen3_5":
                self._size_qwen_cache()
            if not getattr(self.hw_plan, "fits", True) and self.device != "cpu":
                # Refuse loudly instead of OOMing after the first prefill chunk. The
                # streaming skeleton alone (ring + workspace + tables) does not fit.
                raise RuntimeError(
                    f"not enough VRAM for this checkpoint: {self.hw_plan.reason}. "
                    "The engine needs a card that can hold at least the streaming ring "
                    "plus the tilelang workspace; see docs/README.md.")
        if expert_store is None and self.tier.level >= OptLevel.BALANCED:
            # production tiers own their ExpertStore (RAM cache + VRAM slots)
            try:
                from .loading import ExpertStore
                # elastic RAM budget: size the expert cache from what is actually free,
                # so other agents/processes on the box do not cause MemoryError
                try:
                    import psutil
                    avail = psutil.virtual_memory().available
                    cache_budget = int(avail * 0.5)  # larger RAM cache = fewer disk misses in fetch
                except Exception:
                    cache_budget = int(self.tier.expert_cache_gb * (1 << 30))
                hw = self.hw_plan
                _slots = int(__import__("os").environ.get(
                    "TI_SLOTS", hw.expert_slots if hw is not None else 96))
                # A card with expert slots serves its decode experts through Marlin, whose
                # repack is 43 us against a 20 us GEMV. Size the repack cache to the slot
                # count so each expert is interleaved once instead of once per token.
                # Zero slots (this 8 GB box) never reaches Marlin on decode, and the
                # setter leaves the conservative default there.
                if _slots > 0:
                    try:
                        from .deepseek.gemv_fp4_marlin import set_cache_capacity

                        set_cache_capacity(_slots)
                    except Exception:
                        pass
                from .budget import budget as _host_budget
                _hb_room = _host_budget().room()
                # 没有硬件计划时也不设绝对上限：按当前可用的宿主预算给，
                # 60% 留给缓存（剩下的留给层 slab 和运行时）。16GB 这个数是
                # 47GB 内存机器上的观察值，不该是所有机器的默认。
                _plan_cache = (hw.cache_bytes if hw is not None
                               else max(512 << 20, int(_hb_room * 0.6)))
                # leave the majority of what is left to the pinned layer slab, which
                # _flat_setup requests later: 45% here, the rest there
                # The layer slab asks for its bytes AFTER this, so the cache may only
                # take a minority of what is free or the box starts paging to C: and
                # every read turns into a swap-in (observed: 9.3s/token).
                # Size the host expert cache from what is genuinely free after the pinned
                # layer slab, not from a flat fraction: the slab needs (n_layers -
                # resident) * layer_b and is requested *after* this, so a fraction that is
                # too large starves it (measured: 9.3s/token with a 15.3GB cache on this
                # box). TI_CACHE_GB overrides for A/B.
                # THE host pool: claim our whole share of physical RAM ONCE, as pinned
                # (DMA-able) memory, and derive every buffer from what the OS actually
                # handed over. The slab and the expert cache are both carved out of this
                # one range. Sizing from an estimate and asking again later is what
                # created a second pool and a CUDA OOM; the claimed amount is the only
                # number that is true. Other programs may be pushed out of RAM -- that
                # is the accepted trade -- but the model itself must never page, because
                # a paged expert read costs 10x a hit.
                _budget = _host_budget()
                _avail = _budget.available()
                # The headroom is not slack. It has to cover, simultaneously:
                #   - the block build that runs right after this claim: every streamed
                #     layer's host weights are read (~430MB/layer, ~13GB for 30 layers)
                #     before the flat slab copies them into this pool and releases them;
                #   - a fresh pinned staging block for pageable->GPU copies (engram,
                #     embed), which fails with driver OOM once the pool has eaten the box;
                #   - the OS and other applications.
                # Measured failure: pool at avail-10GB (24GB) + 13GB of weights -> swap.
                # Fill physical RAM: a full pool is the design goal (cache hits beat
                # disk reads). The reserve only covers the start-up build (resident
                # layers' host temporaries are freed as they move to VRAM) and OS slack.
                # cudaHostAlloc succeeds up to ~29GB on this box; above that the driver
                # refuses the whole request and the halving retry lands at 15GB (which
                # halves the expert cache and doubles decode time). Cap it.
                # Size the pool from what is free and let _commit() shrink it: that path
                # already retries at half size when the driver refuses (pinned allocations
                # have a ceiling well below physical RAM), and the caches are sized from
                # the bytes actually handed over. Hand-tuning a cap here only traded
                # hit rate for a guess.
                # 上限原来写死 28GB（笔记本上 pin 池的实测天花板）。在 371GB 内存的
                # 机器上等于把 300GB 闲置：池子只有 28GB，缓存再大也装不下，
                # 每个 token 的专家读全靠这 28GB 兜。改成按可用内存取：
                # 留 5GB 给进程自身，再留 25% 不动（给页缓存和内核），其余全给池。
                # TI_HOST_POOL_GB 仍然可以硬指定。
                # 用 total 而不是 available：available 已经把页缓存扣掉了，而读模型
                # 权重会在 page cache 里留下几百 GB —— 那些页是可回收的，池子要用的
                # 时候内核会把它们让出来。按 available 算等于让模型自己的读缓存把
                # 专家缓存挤掉（实测 371GB 的机器上池子只拿到 151GB）。
                # 只保留 15% 的绝对余量给内核和其它进程。
                # **池子上限必须同时受「物理总量」和「此刻真的能用」两条约束。**
                #
                # 只按物理总量算（total - 15%）会忽略桌面与其它进程已经占走的部分：
                # 本机 47.2GB，桌面占 14GB，可用只剩 32.8GB，按 40.1GB 去 claim 的
                # 缺口只能落到 pagefile —— 而 pinned 页落进 swap 意味着每次专家访问
                # 变成 swap-in（比读盘还慢），大块上 cudaHostRegister 会直接失败
                # （表面症状是 cuMemSetAccess OOM，而显存还有 2.6GB 空闲）。
                #
                # 也不能只用 available()：它对页缓存敏感，模型自己读权重留下的页缓存
                # 会把 available 压得很低（另一台机器上实测只给出 151GB）。
                # 取两者的小值：available 扣掉启动期必需余量，再和 total 的 85% 比。
                _tot = int(getattr(_budget, "total", 0) or 0)
                _by_total = (_tot - int(_tot * 0.15)) if _tot else (1 << 62)
                # 启动期余量：streamed 层权重（约 430MB/层）、engram/embed 的 pinned
                # staging 块、操作系统与其它进程。
                _by_avail = int(_avail) - (6 << 30)
                _base = min(_by_total, _by_avail)
                _target = max(192 << 20,
                              min(_budget.room(), _base))
                _pov = __import__("os").environ.get("TI_HOST_POOL_GB")
                if _pov:
                    # An explicit override wins over the RAM-derived cap. It is the knob
                    # used to probe how large the pinned pool can grow before the driver
                    # refuses (cuMemSetAccess), and a cap that can only shrink cannot
                    # answer that.
                    try:
                        _o = max(192 << 20, int(float(_pov) * (1 << 30)))
                        _target = min(_budget.room(), _o)
                    except ValueError:
                        pass
                _taken = int(_budget.claim(_target, real=True))
                from .budget import HostPool
                from .loading import BlobPool

                _pool_view = getattr(_budget, "_claim_buf", None)
                self.host_pool = (HostPool.from_claimed(_pool_view, _taken)
                                  if _taken and _pool_view is not None else None)
                # Reserve the slab's share first: _flat_setup runs on the first forward
                # and carves its per-layer pinned blocks out of this same pool, so the
                # cache must not eat bytes the slab is about to need.
                # Size the slab reserve from the planner's own per-layer number: the weights
                # are not loaded yet, so this is the only honest figure available, and the
                # old flat 28% of RAM reserved 13.2GB for a 6.5GB slab, starving the cache.
                # **每层 streaming 字节用规划器的实测值，不能写死 178MB。**
                #
                # 178MB 是 safetensors header 里记的 fp8 载荷，而一层真正要搬进
                # 环形缓冲的是一块连续 buffer，按最坏层开 463MB（本机实测）。
                # 32 层差 9GB，slab 预留不够 → flat slab 建不起来 → 退回逐张量
                # pinned 拷贝 → 那批临时内存 + 专家缓存一起把宿主池吃穿，
                # 首次 forward 直接 OOM（表面症状是 slot arena 的 !handles_ 断言）。
                # **slab 预留按「每层实际字节」，不是「最坏层字节」。**
                #
                # 上一版用了 max_layer_bytes（最坏层 468MB），33 层 x 1.15 = 18.7GB 的
                # 预留把整个池子吃光，缓存只剩 1.24GB（66 个专家，对每步 240 个需求
                # 覆盖 27%）。而 slab 实际每层只吃 237MB —— 实测装完 33 层共 7.83GB。
                # 两者差 2.4 倍：预留是按最坏层算的，装载是按实际字节算的。
                #
                # max_layer_bytes 的正确用途是「一个环形缓冲要开多大」（最坏层决定
                # 单个 buffer 尺寸，必须按最坏的来），不是「所有层的总量」（那是
                # 各层实际字节求和）。把这两个概念混在一起就是过度预留。
                # 数值取 240MB：这是 33 个 streamed 层「实际字节求和 / 层数」的实测均值
                # （slab 装完 33 层共 7.83GB），不是 header 报的 172MB 平均。
                # 两者差 40%：header 的 layer_bytes 是各层 tensor 字节的平均，
                # 而 slab 按 sizes[i]（每层实际 span）分配，含对齐和额外 buffer。
                # 按 172MB 预留时 slab 差 2GB 建不起来，池子被 cache 独占
                # （实测 pool used=21.57/21.57, free=0.00）。
                _per_layer = 240e6
                try:
                    _lb = float((_prof or {}).get("layer_bytes", 0) or 0)
                    if _lb > 0:
                        _per_layer = max(_lb, float((_prof or {}).get("max_layer_bytes", 0) or 0) * 0.5)
                except Exception:
                    _per_layer = 240e6
                _slab_need = (self.cfg.n_layers
                              - (hw.gpu_layers if hw is not None else 0)) * _per_layer
                _slab_reserve = 0
                _flat_flag = __import__("os").environ.get("TI_FLAT")
                if self.host_pool is not None and not (_flat_flag is not None
                                                       and _flat_flag != "1"):
                    # only reserve slab room when flat streaming may use it; with
                    # TI_FLAT=0 the slab never runs and the bytes belong to the cache
                    # The slab itself is exactly sum(streamed layer bytes) (5.3GB here);
                    # the rest of the reserve is the build window (a layer is read on the
                    # host, copied into the pool, then released) plus slack. 3.5x reserved
                    # 18.7GB and left the expert cache with 9-13GB, which holds ~600 of the
                    # 1439 experts ten decode steps touch -- that is why the hit rate was
                    # 50%. 1.6x leaves the cache ~19GB while still covering the build.
                    # 1.1x: the slab is exactly sum(streamed layer bytes); the extra 10%
                    # covers the build window. 1.6x left the pool too short for the slab
                    # itself (measured "slab unavailable" -> the per-tensor fallback, which
                    # then needed a bigger prefill workspace and OOM'd).
                    # Reserve what the slab actually needs, not as much as the pool will
                    # give: the old upper bound was (free - 192MB), so on a pool that is
                    # both cache and slab the two reservations together could exceed it --
                    # observed as cache 15.6GB + reserve 6.7GB against a 20.8GB pool, which
                    # left the slab 0 bytes and pushed all 30 streamed layers onto the
                    # per-tensor copy path (1167 DMAs per decode step, 458ms of GPU time).
                    _slab_reserve = min(int(_slab_need * 1.15) + (1 << 30),
                                        max(0, self.host_pool.free - 192 * (1 << 20)))
                    # whatever the cache does not claim must stay free for the slab
                    if self.host_pool is not None:
                        _slab_reserve = min(_slab_reserve,
                                            max(0, self.host_pool.free - (192 << 20)))
                _cache_pool = None
                _cache_bytes = max(192 << 20, int(_plan_cache))
                _ov = __import__("os").environ.get("TI_CACHE_GB")
                _ov_bytes = None
                if _ov:
                    try:
                        _ov_bytes = int(float(_ov) * (1 << 30))
                    except ValueError:
                        _ov_bytes = None
                if _ov_bytes is not None:
                    _cache_bytes = _ov_bytes
                if self.host_pool is not None:
                    _pool_avail = max(192 << 20, self.host_pool.free - _slab_reserve)
                    # the override may not exceed what the pool can actually guarantee:
                    # capacity larger than the backing slab makes every take() fail and the
                    # read path silently falls back (measured: TI_CACHE_GB=14 with a 6GB
                    # slab -> 555 fallbacks, decode 1.1s -> 1.9s)
                    # Hard ceiling on the host cache. Two costs trade here: a bigger cache
                    # keeps decode off the SSD (40 layers x top-6 = 240 experts/token, so
                    # a small claim spills and the direct reader showed ~180ms/token),
                    # while a smaller one leaves the pool enough for the full 30-layer
                    # slab (5.3GB) -- below that the slab comes back partial and the
                    # remaining layers pay 30 per-tensor copy_ calls per token.
                    # Measured at the decode step: 12GB cache + slab 30/30 = 1.58 tok/s,
                    # 16GB cache + slab 10/30 = 1.61 tok/s. Cache wins by a hair.
                    # Cache vs slab: they share the pinned pool and are built in this
                    # order, so whatever the cache claims here is unavailable to the slab
                    # below. The slab is worth more per GB -- a slab layer replaces its
                    # 178MB of per-tensor H2D every token, while a cache slot only avoids
                    # a read when the same expert is reused. Measured: 16GB cache leaves
                    # the slab 8 of 30 layers and decode sits at 519ms/step with 88% hit;
                    # 12GB lets the slab cover every streamed layer. TI_CACHE_CAP_GB
                    # exposes the split.
                    # 12, not 16. Measured sweep on this box (8GB VRAM, 47GB RAM):
                    #   cap=16  814 ms   0/30 layers on the slab
                    #   cap=12  784 ms   4/30
                    #   cap=10  785 ms  14/30
                    #   cap= 8  825 ms  24/30   (cache starts missing too often)
                    #   cap= 6 1059 ms    all
                    #   cap= 4 1222 ms    all
                    # The slab ends the per-tensor copy path (1167 DMAs -> one per layer
                    # per step) but the cache prevents reads, so there is an interior
                    # optimum; 12 and 10 tie, and 12 leaves more of the pool free for the
                    # layer weights the build has not read yet.
                    # 缓存上限不再写死 12GB：那是 8GB 显存 + 47GB 内存的笔记本上扫出来的
                    # 内部最优（缓存和层 slab 争同一块 pin 池），搬到 371GB 内存的机器上
                    # 等于把 300GB 内存闲置，每个 token 的专家读全靠 12GB 的缓存兜。
                    # 改成按宿主预算自适应：不给显式覆盖时，上限 = 预算里除 slab 之外
                    # 能拿到的全部（_pool_avail 已经扣掉了 slab_reserve），再留 25% 余量
                    # 给尚未加载的层权重。TI_CACHE_CAP_GB 仍然可以硬指定。
                    _cap_env = __import__("os").environ.get("TI_CACHE_CAP_GB")
                    if _cap_env is None:
                        # 0.75 的余量是给「还没加载的 streamed 层权重」留的。
                        # 当**所有层都常驻**（L20 的 40/40）时根本不存在这件事，
                        # 那 25% 就是白留：实测 L20 上池子只拿到 231GB，
                        # 而模型全部 15360 个专家要 268.9GB —— 差 14%，预热永远装不完，
                        # decode 每步还得读 13-25 次盘。
                        # 所以：没有 streamed 层时只留 6% 给分配器抖动。
                        _n_stream = 0
                        try:
                            _n_stream = int(len(self._stream_layers()))
                        except Exception:
                            _n_stream = 0
                        _frac = 0.94 if _n_stream == 0 else 0.75
                        _cap_bytes = max(192 << 20, int(_pool_avail * _frac))
                    else:
                        _cap_bytes = max(192 << 20, int(float(_cap_env) * (1 << 30)))
                    _pool_avail = min(_pool_avail, _cap_bytes)
                    _cache_bytes = min(_ov_bytes, _pool_avail) if _ov_bytes else _pool_avail
                    _slot = self.host_pool.alloc(_cache_bytes)
                    if _slot is not None:
                        _cache_pool = BlobPool(_slot, _cache_bytes)
                self.expert_store = ExpertStore.from_store(
                    self.store,
                    slot_arena=bool(getattr(self.tier, "slot_arena", False)),
                    # parallel reads saturate the device around QD=8-16; scale with the
                    # core count instead of pinning a machine-specific thread count
                    # 并行读深度：跟随 CPU 数（pread 在 C 侧释放 GIL，线程数真的
                    # 决定并行度），不再写死 16 的上限。
                    reader_threads=int(__import__("os").environ.get("TI_READERS", "0")) or 0,
                    cache_bytes=_cache_bytes,
                    pool=_cache_pool,
                    # LRU, not ARC. Measured on this model's real routing over three
                    # runs: lru 0.789/0.789/0.788 hit and 30.9-31.3 GB read, arc
                    # 0.710/0.685/0.667 and 34.8-39.4 GB, s3fifo 0.567. ARC's ghost lists
                    # are tuned for a workload with a stable hot set to promote; this
                    # model's routing is flat enough that they mostly cost capacity.
                    policy=__import__("os").environ.get("TI_CACHE_POLICY", "lru"),
                    vram_slots=_slots if device != "cpu" else 0,
                    device=device if device != "cpu" else "cpu",
                    # direct=NO_BUFFERING reads: measured 6.3GB/s at 6 threads vs
                    # 0.26GB/s through the Windows file cache (which serialises the
                    # multi-threaded pread path on a big-blob stream)
                    direct=True,
                )
                # Multi-GPU: hand the store the card list so it can spread its slot
                # arenas over them in proportion to free VRAM. The store keeps whole
                # slots per card, so a fetch lands on one device and the experts kernels
                # read it from there; on a single card this list has one entry and the
                # allocation is byte-for-byte the old one.
                if self.devices and self.expert_store is not None:
                    try:
                        self.expert_store.set_devices(
                            [f"cuda:{d.index}" for d in self.devices],
                            [int(d.free) for d in self.devices])
                    except Exception as _de:
                        print(f"[tinyinfer] expert multi-device skipped ({_de!r})",
                              flush=True)
            except Exception as e:
                print(f"[tinyinfer] ExpertStore disabled: {e}")
                self.expert_store = expert_store
        else:
            self.expert_store = expert_store
        # The VRAM slot pool is faster but not bit-reproducible yet (a slot can be
        # re-targeted while the GEMV that reads it is still queued; docs/README.md
        # section 10). TI_NO_SLOTS=1 trades about 40% of decode for deterministic logits.
        # **冷启动 = 进程一起来就让磁盘开工，并且请求不被它挡住。**
        #
        # 两件事同时成立，缺一个都不对：
        #   1) 磁盘从启动那一秒就不该闲着 —— 后台线程持续把专家读进 RAM，
        #      没有请求时也读，读到装不下为止（覆盖式回收，不主动释放）；
        #   2) 请求不排队等它 —— 请求要哪个专家就自己直接读哪个，
        #      不等「预热队列轮到它」，也不等预热整体跑完才开始接请求。
        #
        # 两者共享磁盘：请求是最高优先级，预热填请求之间的空隙（`_busy` 让路）。
        # 所以默认开启（TI_WARM=0 可关），而不是等第一个请求才动手 ——
        # 那会把启动后的空闲期白白浪费掉。
        _warm_env = __import__("os").environ.get("TI_WARM", "1")
        if (self.expert_store is not None
                and device != "cpu"
                and _warm_env != "0"):
            try:
                from .expert_tiers import ExpertTierManager

                _nl = int(getattr(self.cfg, "n_layers", 40))
                _ne = int(getattr(self.cfg, "n_routed_experts", 0) or 384)
                # 分级管理器由引擎创建（整机只有一个），MoE 通过 store 拿同一个。
                _mgr = self.expert_store._tier_mgr = ExpertTierManager(
                    self.expert_store, n_layers=_nl, n_experts=_ne,
                    device=str(device), warm=False)
                if _mgr.start_warm():
                    print(f"[tinyinfer] warming {_nl}x{_ne} experts into RAM "
                          f"(background, not on the critical path)", flush=True)
                # 把「这套配置够不够」直接说出来。这是用户判断该不该调 topk 的
                # 唯一依据 —— 引擎不替他降质量，但必须告诉他现状：
                # 一层的并集装不下时，decode 的每一步都在搬专家。
                _cap = _mgr.capabilities()
                # 用「一步 decode 需要的专家数」（n_layers*topk=240），
                # 不是「全部专家」（15360）—— 后者没人能达到，说出来没有意义。
                _need_step = _cap["slots_needed_a_step"]
                _have = _cap["slots"]
                if _have and not _cap["can_hold_a_step"]:
                    _pct = 100.0 * _have / max(1, _need_step)
                    print(f"[tinyinfer] expert slots cover {_pct:.0f}% of one decode "
                          f"step ({_have}/{_need_step}); every step re-fetches experts. "
                          f"TI_TOP_K=N can trade quality for IO here", flush=True)
                elif _have:
                    print(f"[tinyinfer] expert slots cover a full decode step "
                          f"({_have}/{_need_step}); steady-state fetch is zero", flush=True)
            except Exception as _we:
                print(f"[tinyinfer] warm-up skipped ({_we!r})", flush=True)
        if __import__("os").environ.get("TI_NO_SLOTS") == "1":
            for _st in (getattr(self, "expert_store", None),):
                try:
                    if _st is not None:
                        _st.slots = None
                except Exception:
                    pass
        self.max_layers = self.cfg.n_layers if max_layers is None else min(max_layers, self.cfg.n_layers)
        self.allow_missing_layers = allow_missing_layers
        self.prefill_chunk = 2048  # overridden by the hardware plan on cuda
        # TI_PREFILL_CHUNK=1 forces the exactly reference-matching prefill path (each
        # token its own block); the planner's chunk is ~0.5% harder but 1.6x faster
        # (docs/README.md 13).
        _pc = __import__("os").environ.get("TI_PREFILL_CHUNK")
        if _pc:
            try:
                self.prefill_chunk = max(1, int(_pc))
            except ValueError:
                pass
        # how many layers may stay resident on the device; the rest stream per forward
        if gpu_layers is not None:
            self.gpu_layers = gpu_layers
            self.head_gpu = gpu_layers <= 20  # keep the head off-card when many layers are resident
            if head_on_cpu:
                self.head_gpu = False
        elif device == "cpu":
            self.gpu_layers = 40
            self.head_gpu = False
        else:
            # hardware-derived plan (resources.py): adaptive VRAM/RAM budgeting for any
            # card size (8GB laptop .. 80GB desktop .. unified-memory boxes), no manual flags.
            hw = self.hw_plan
            import os as _os
            # overrides for experiments: residency 0 makes every layer stream through
            # the flat pair, which is more H2D but frees the VRAM the pair needs
            self.gpu_layers = max(0, int(_os.environ.get("TI_GPU_LAYERS", hw.gpu_layers)))
            # the plan already accounted for the head when it sized the resident set;
            # re-deciding here with a fixed layer threshold put the head on the host even
            # when the plan kept it on the card, and _head_logits then sliced a host head
            # against device activations
            self.head_gpu = not hw.head_on_cpu
            self.prefill_chunk = int(_os.environ.get("TI_CHUNK", hw.prefill_chunk))
            # placement: while most layers stream, small activations round-trip every
            # op; keep HC-style tiny ops on the host (big-VRAM boxes stay all-GPU).
            from .placement import small_op_on_cpu
            # hc_post on the host costs one D2H sync per call (four per layer). At M=1
            # that is cheap because the GPU queue is short; at M>1 the sync waits for
            # the whole layer (measured: 4 x 150ms x 38 layers = 24s of a 29s forward).
            self.small_ops_on_cpu = (small_op_on_cpu(self.gpu_layers, self.max_layers)
                                     and __import__("os").environ.get("TI_HC_GPU") != "1")
        if self.device != "cpu" and self.tier.level < OptLevel.AGGRESSIVE:
            # Reference-project behaviour: put the whole model on the device, like vLLM --
            # no streaming, no partial residency. The original comment assumed "if it does
            # not fit, the OOM is the answer", but on Windows WDDM does not fail: it pages
            # the overflow out to system RAM and keeps going. A 40-layer run on this 8GB
            # card happily reported 13.27GB allocated against 7.93GB of VRAM and ran at
            # half speed with no error at all, which is the worst of both worlds.
            # So check the arithmetic up front and refuse, the way a reference stack
            # would, instead of silently paging.
            _per_layer = 0
            try:
                _per_layer = int((getattr(self, "_model_prof", None) or {}).get("layer_bytes", 0))
            except Exception:
                _per_layer = 0
            # The reference stacks keep the WHOLE checkpoint resident -- every layer and
            # every routed expert. For DeepSeek-V4.1-Flash that is 40 layers of ~164MB
            # plus 40 x 384 experts of ~18MB (~281GiB), which is why this tier needs a
            # machine with hundreds of GiB of device memory. Counting only the layers
            # would have let an 8GB card "pass" a check that cannot possibly fit.
            _n_exp = 0
            _exp_b = 0
            try:
                _p = getattr(self, "_model_prof", None) or {}
                _n_exp = int(getattr(self.cfg, "n_routed_experts", 0) or 0)
                _exp_b = int(_p.get("expert_bytes", 0))
            except Exception:
                _n_exp = _exp_b = 0
            _expert_total = self.max_layers * _n_exp * _exp_b
            if _per_layer > 0:
                _need = self.max_layers * _per_layer + _expert_total
                _free = 0
                try:
                    _free, _total = torch.cuda.mem_get_info()
                except Exception:
                    _free = 0
                # Leave room for the KV cache, the head and the prefill workspace. A flat
                # 8% was not enough: 6.4GiB of layers against 6.9GiB free passed the test
                # and still had only ~0.5GiB for a KV cache that needs several GiB on a
                # long context. Reserve 2GiB outright, which keeps the check honest on a
                # small card while the large cards this tier targets are unaffected.
                _budget = int(_free - 2 * 2**30)
                if _need > _budget:
                    raise RuntimeError(
                        f"tier {self.tier.level.name} keeps the whole checkpoint resident "
                        f"(the reference behaviour: no streaming), and that needs "
                        f"{_need / 2**30:.0f} GiB "
                        f"({self.max_layers} layers x {_per_layer / 2**20:.0f} MiB"
                        f"{f' + {self.max_layers * _n_exp} experts' if _expert_total else ''}) "
                        f"against {_free / 2**30:.1f} GiB free. Windows would page the "
                        f"overflow instead of failing, which runs at roughly half speed. "
                        f"Use OptLevel.AGGRESSIVE, which streams layers and experts.")
            self.gpu_layers = self.max_layers
            print(f"[tinyinfer] tier {self.tier.level.name}: all {self.max_layers} layers "
                  f"resident (no streaming)", flush=True)
        self._blocks: dict[int, Block] = {}
        self._last_routing: dict[int, list[int]] = {}
        # expert prefetch is opt-in: with the RAM cache holding ~600 of 15360 experts
        # the prediction misses often enough that the prefetch and the fetch read the
        # same blob twice (measured 5.3GB/step against 3.5GB of unique data)
        # tier.prefetch is the policy, TI_PREFETCH is the override: the field used to be
        # dead, so a tier that asked for prefetching only got it if the env var happened
        # to be set. Now the tier decides unless the operator forces it.
        _pf_env = __import__("os").environ.get("TI_PREFETCH")
        if _pf_env is None:
            self.enable_prefetch = bool(getattr(self.tier, "prefetch", False))
        else:
            self.enable_prefetch = bool(int(_pf_env))
        self.cache_wo_a = False
        self.profile = False
        # CPU-side learned expert prefetcher (context tables; ~95% hit rate measured)
        self.predictor = None
        if self.tier.level >= OptLevel.BALANCED:
            from .routes import RoutePredictor
            self.predictor = RoutePredictor()
        if tokenizer is not None and self.cfg.engram_layer_ids:
            # building the compressed token map walks the whole 129k vocab: only do it when
            # the engram tables are actually on disk
            shards = {self.store.shard_of(f"layers.{i}.engram.embed.weight") for i in self.cfg.engram_layer_ids}
            if all((self.store.model_dir / s).exists() for s in shards):
                from .deepseek.engram import EngramHashes
                try:
                    self.hashes = EngramHashes(self.cfg, tokenizer)
                except Exception as e:  # weighted table missing / tokenizer mismatch: run without engram
                    print(f"[tinyinfer] engram disabled: {e}")
        # The embedding table is 1.32GB of VRAM for a 10KB row lookup per token. On a
        # card that cannot hold every layer it is pure waste: keeping it in host RAM
        # frees enough for two resident layers plus the expert staging pool.
        # TI_EMBED_GPU=1 forces the old behaviour (table in VRAM); it is the escape hatch
        # when a new heuristic regresses a machine that used to run.
        # Tensor parallel: one process holds every rank as a thread in Flare; here the
        # caller launches one process per rank and we join the default process group.
        self.tp = None
        self.tp_ranks = int(getattr(self.tier, "tp_ranks", 1) or 1)
        if self.tp_ranks > 1 and device != "cpu":
            import os as _os

            import torch.distributed as _dist

            if not (_dist.is_available() and _dist.is_initialized()):
                _dist.init_process_group(
                    backend="nccl",
                    rank=int(_os.environ.get("RANK", "0")),
                    world_size=self.tp_ranks,
                )
            from .tp import Shard

            self.tp = Shard(_dist.get_rank(), self.tp_ranks)
            print(f"[tinyinfer] tensor parallel rank {self.tp.rank}/{self.tp.world}", flush=True)
        # embed 表放哪里必须与规划器一致。
        # 这里曾是硬编码 `self.embed_on_cpu = False`（理由：CUDA graph 捕获期禁止
        # D2H），但 `resources.plan` 按 `embed_on_cpu=True` 记账 —— 于是规划器
        # 少算 1.23GB，8GB 卡上多给了 5 层常驻，40 层模型连槽位 arena 都分配
        # 不出来（!handles_.at(i) INTERNAL ASSERT）。
        # 现在按规划结果走：规划说 head 在 GPU、embed 不在，就照做。
        # TI_EMBED_GPU=1 可强制放回显存（捕获图时需要）。
        _plan_embed_gpu = bool(getattr(self.hw_plan, "embed_on_cpu", False) is False
                               and self.hw_plan is not None)
        self.embed_on_cpu = (not _plan_embed_gpu)
        if __import__("os").environ.get("TI_EMBED_GPU") == "1":
            self.embed_on_cpu = False
        # embed / head / norm are 100MB-1.3GB allocations: route them through the
        # claimed pool too, so no large buffer ever appears outside it.
        if getattr(self, "host_pool", None) is not None:
            self.store.build_pool = self.host_pool
        # names come from the adapter: a checkpoint decides what it calls these (DeepSeek
        # keeps embed/norm/head at the top level, Qwen nests them under
        # model.language_model and ties the head to the embedding)
        # embed(1.3GB) / norm / head(1.3GB) 是启动时最重的三次读盘，互不依赖。
        # 串行读它们就是启动阶段单核 100% 的来源之一（py-spy 停在 ShardStore.get）。
        from concurrent.futures import ThreadPoolExecutor as _TPE3
        _keys = [self.adapter.embed_key, self.adapter.norm_key]
        _vals = {}
        def _rd(_k):
            return _k, self.store.get(_k)
        with _TPE3(max_workers=2, thread_name_prefix="tinyinfer-hdr") as _ex3:
            for _k, _v in _ex3.map(_rd, _keys):
                _vals[_k] = _v
        _emb = _vals[self.adapter.embed_key]
        self.embed = _emb if self.embed_on_cpu else _emb.to(device)
        self.out_norm = _vals[self.adapter.norm_key].to(device)
        # the output head is the largest single tensor: off-VRAM it costs ~40ms/token but
        # frees 1.3GB. A tied checkpoint has no separate head, so reuse the embedding.
        head_key = self.adapter.head_key
        # only a separate head can be pushed off the card: a tied one IS the embedding,
        # so it has to live wherever the embedding does
        self.head_on_cpu = (head_key is not None and device != "cpu"
                            and not self.head_gpu)
        if head_key is None:
            self.head = self.embed
        elif self.head_on_cpu:
            self.head = self.store.get(head_key)
        else:
            _hd = self.store.get(head_key).to(device)
            # 这个 checkpoint 的 head 和 embed 是**同一张表的两份拷贝**（形状
            # 都是 129280x5120，内容一致）。各存一份白占 1.23GB —— 实测 8GB 卡上
            # 初始化后 embed 1.233GB + head 1.233GB = 2.466GB，占掉了四成显存，
            # 直接导致 40 层模型连槽位 arena 都分配不出来。
            # 用形状 + 前 4KB 字节判断是不是同一张表：不做全量比较（那是 1.2GB
            # 的读），只做一个足够强的指纹。
            _same = False
            try:
                _e = self.embed
                if (torch.is_tensor(_e) and _e.shape == _hd.shape
                        and _e.dtype == _hd.dtype and _e.device == _hd.device
                        and _e.numel() > 0):
                    _n = min(4096, _e.numel())
                    _same = bool(torch.equal(_e.view(-1)[:_n], _hd.view(-1)[:_n]))
            except Exception:
                _same = False
            if _same:
                # 复用 embedding：省 1.23GB，而且两者本来就该一致
                self.head = self.embed
                del _hd
            else:
                self.head = _hd
        self.eps = self.cfg.rms_norm_eps
        # Tier >= 2 serves with speculation: load the DSpark stages now so the first
        # request does not pay for them. A missing/incompatible set disables MTP for
        # the session instead of failing the engine.
        _mtp_ok = bool(getattr(self.tier, "mtp", False)) and bool(
            getattr(self.hw_plan, "speculation", True))
        # MTP 的 draft 需要 [2, T, 512] 的拼接，而当前 serve 路径在 T 变化时给出了
        # [1, T, 512]，服务端直接 500。默认关掉，等 shape 修好再开。
        if __import__("os").environ.get("TI_MTP", "0") == "1":
            # force the draft stages on even when the plan says they do not fit: used to
            # measure the accept rate on a card the planner keeps them off for
            _mtp_ok = True
        if getattr(self.tier, "mtp", False) and not _mtp_ok:
            print("[tinyinfer] MTP off: the draft stages do not fit beside the streaming "
                  "ring on this card (see the plan above)", flush=True)
        # MTP 的 draft 需要 [2, T, 512]，而当前 serve 路径在 T 变化时给出 [1, T, 512]，
        # 服务端直接 500。默认不加载（它也是启动阶段的一串逐 tensor 读盘）；
        # shape 修好后用 TI_MTP=1 打开。
        if _mtp_ok and device != "cpu" and __import__("os").environ.get("TI_MTP", "0") == "1":
            try:
                if self.load_mtp():
                    print("[tinyinfer] MTP draft stages loaded (tier %s)" % self.tier.level.name,
                          flush=True)
            except Exception as exc:
                print(f"[tinyinfer] MTP unavailable: {exc}", flush=True)

        # Allocate EVERYTHING now: blocks, host weights, pinned slab. Blocks used to be
        # constructed lazily on their first forward, so RAM climbed for a minute after
        # the engine was "up" -- and because the pool was already claimed, that climb
        # ended in swap. One build here makes the footprint final before the engine is
        # handed out; nothing is allocated during generation except the expert cache,
        # which fills inside the pool.
        self._prebuild()

    # -- start-up build ----------------------------------------------------
    def _prebuild(self) -> None:
        """Build every block and the pinned slab NOW, not on the first forward.

        Each block reads its host weights from disk (~430MB per streamed layer). Doing
        that lazily during the first request is what made RAM climb after start-up;
        because the pool is already claimed, the climb ended in swap (measured: 24GB
        pool + 13GB weights + 13GB OS = 50GB). Pay the whole cost here so the footprint
        is final before the engine is handed out.
        """
        if self.device == "cpu" or getattr(self, "_prebuilt", False):
            return
        self._prebuilt = True
        if self.layer_ids is not None:
            layers = list(self.layer_ids)
            if self.allow_missing_layers:
                avail = self.available_layers()
                layers = [i for i in layers if i < avail]
        else:
            n = self.max_layers
            if self.allow_missing_layers:
                n = min(n, self.available_layers())
            layers = list(range(n))
        if not layers:
            return
        # No _warm_weights here: it read all 17GB into private memory (plus a bytearray
        # copy per tensor) only to page-cache it, and that transient is what pushed
        # commit to 90GB. Blocks read their own weights; the OS cache warms itself.
        streamed_set = self._stream_layers()
        pool = getattr(self, "host_pool", None)
        try:
            if pool is not None:
                # weights are read straight into pool slices from here on: the build
                # happens INSIDE the claimed pool, so Task Manager shows the pool's one
                # jump and nothing after it
                self.store.build_pool = pool
            # 逐层串行构建 = 启动时单核 100%（实测 64 核机器上只有 C0 在动）。
            # 每层是独立的：读自己的分片 + 建自己的子模块，互不依赖，所以按 CPU 数
            # 并发构建。self.block() 内部写 self._blocks，加锁保护那一次写入即可。
            _build = [i for i in layers if i not in streamed_set]
            if len(_build) > 1:
                import os as _os2
                from concurrent.futures import ThreadPoolExecutor as _TPE2
                # dict 写的是不同 key，CPython 下由 GIL 保证原子，不需要额外锁
                _nw = max(2, min(len(_build), (_os2.cpu_count() or 4)))

                def _mk(i):
                    b = self.block(i)
                    # resident layers live in VRAM for the whole run; the pinned host
                    # copy only exists for a to("cpu") that never happens on them
                    b._cpu_refs = None
                    return b

                with _TPE2(max_workers=_nw, thread_name_prefix="tinyinfer-build") as _ex2:
                    list(_ex2.map(_mk, _build))
            else:
                for i in _build:
                    b = self.block(i)
                    b._cpu_refs = None
            # flat streaming builds each streamed block and copies its weights into the
            # pinned slab -- all inside the same pool
            try:
                self._flat_enabled()
            except Exception as exc:
                print(f"[tinyinfer] flat setup at start-up failed: {exc!r}", flush=True)
        finally:
            self.store.build_pool = None
            if self.device != "cpu":
                # Compile the tilelang kernels NOW. The first user request otherwise
                # paid ~90s and +6.8GB of compiler memory inside the forward -- the
                # exact "climb after start-up" that must not happen. A one-token
                # forward compiles every kernel the real runs use.
                # OFF by default: a warm-up forward leaves state that reset() cannot
                # fully undo (measured: the first real request then answered an
                # unrelated question). Compilation is paid by the first request instead;
                # that is a JIT cost, not an allocation the pool could own.
                if __import__("os").environ.get("TI_WARMUP", "0") == "1":
                    try:
                        with torch.inference_mode():
                            self.forward(torch.tensor([[0]], device=self.device), 0)
                        self.reset()
                    except Exception as exc:
                        print(f"[tinyinfer] warm-up forward failed: {exc!r}", flush=True)
                # hand the allocator's cached blocks back before the first request: the
                # build leaves them behind and a card this tight needs them for KV
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            # 模型（40 层权重 + KV + workspace）现在确实全部就位了，这才是问驱动
            # 「还剩多少显存」的正确时机。按它把专家槽位扩到能吃满为止：换一张
            # 96GB/141GB 的卡不需要改任何参数，槽位数自己会跟上。
            # 自动重算槽位暂不启用：reinit 会先释放旧 arena 再按实测 free 重建，
            # 而 free 里还有 KV/workspace 的余量没扣干净，实测重建后状态数组和
            # arena 对不齐（slots 有值、buffers 归零 → IndexError）。planner 的
            # 静态值虽然保守（44GB 卡上给 1359 个），但实测 10/10 请求成功。
            # 等 _ptrs() 静态化、decode 改成 CUDA graph 之后再回来做这件事。
            if __import__("os").environ.get("TI_AUTO_SLOTS") == "1":
                _sl = getattr(getattr(self, "expert_store", None), "slots", None)
                if _sl is not None and getattr(_sl, "slots", 0) > 0:
                    try:
                        _n = _sl.reinit()
                        if _n:
                            print(f"[tinyinfer] expert slots sized to {_n}", flush=True)
                    except Exception as _exc:
                        print(f"[tinyinfer] slot grow skipped: {_exc!r}", flush=True)

    # -- residency ---------------------------------------------------------
    def available_layers(self) -> int:
        """How many consecutive layers have all their shards fully on disk (size-checked).

        A safetensors header is readable long before the data segments finish
        downloading; loading such a shard segfaults inside CUDA. Compare the file
        size against header + max data offset instead.
        """
        cached = getattr(self, "_avail_layers", None)
        if cached is not None:
            return cached
        import json
        import struct

        def shard_complete(shard: str) -> bool:
            path = self.store.model_dir / shard
            if not path.exists():
                return False
            try:
                with open(path, "rb") as f:
                    n = struct.unpack("<Q", f.read(8))[0]
                    if n > 200 * 1024 * 1024:
                        return False  # truncated/garbage header
                    raw = f.read(n)
                    if len(raw) != n:
                        return False
                    header = json.loads(raw)
                data_end = max(v["data_offsets"][1] for v in header.values() if "data_offsets" in v)
                return path.stat().st_size == 8 + n + data_end
            except Exception:
                return False

        n = 0
        for i in range(self.cfg.n_layers):
            names = [f"layers.{i}.attn.wq_a.weight", f"layers.{i}.ffn.gate.weight",
                     f"layers.{i}.ffn.experts.0.w1.weight"]
            try:
                shards = {self.store.shard_of(nm) for nm in names}
            except Exception:
                break
            if not all(shard_complete(s) for s in shards):
                break
            n = i + 1
        # Parsing every shard header costs ~10ms each and this is called on every
        # forward: 40 headers = 0.35s per decode step (cProfile). The set of layers on
        # disk does not change during a run.
        self._avail_layers = n
        return n

    def _rt_for(self, layer_id: int):
        """Handoff slots for a layer: the nearest kv_source at or below it.

        Chunk-major execution reuses one process singleton because layers run in order;
        layer-major needs a separate instance per source so followers of different
        chunks cannot observe a stale neighbour's state.
        """
        from .deepseek.compressor import SharedAttentionRuntime, shared_attn

        if not getattr(self, "_layer_major_mode", False):
            return shared_attn
        cache = getattr(self, "_rt_cache", None)
        if cache is None:
            self._rt_cache = cache = {}
        if layer_id in cache:
            return cache[layer_id]
        sources = [s for s in self.cfg.kv_source_layer_ids if s <= layer_id]
        src = max(sources) if sources else -1
        rt = cache.get(src)
        if rt is None:
            rt = cache[src] = SharedAttentionRuntime()
        cache[layer_id] = rt
        return rt

    def _gpu_for_layer(self, layer_id: int, nbytes: int = 200 << 20):
        """Which device this layer's weights belong on.

        Booked through Placement on first use, so the cards fill in proportion to their
        real free VRAM without anyone specifying a split. One card means one entry in the
        pool and every layer lands on it -- same call, same bookkeeping, no branch.

        The byte estimate only drives the *booking*, not the copy: if a card runs out, the
        layer falls back to device 0 rather than failing the run, because a suboptimal
        placement still works and a failed startup does not.
        """
        if self.placement is None:
            return self.device
        dev = self._layer_dev.get(layer_id)
        if dev is None:
            try:
                dev = self.placement.place(f"layer.{layer_id}", int(nbytes))
            except Exception:
                dev = 0
            self._layer_dev[layer_id] = dev
        return f"cuda:{dev}" if str(self.device).startswith("cuda") else self.device

    def _expert_route(self):
        """How the routed experts are spread over the cards, or None for one card.

        Built once from the same device list that drives placement, so the expert split
        and the weight split cannot disagree about how many ranks exist. One card means
        nothing to split and this returns None -- the caller then leaves ep_world at 1 and
        the per-layer filter is a no-op, not a special case.
        """
        cached = getattr(self, "_ep_route", None)
        if cached is not None:
            return cached if getattr(cached, "world", 1) > 1 else None
        if self.pool is None or not self.devices or len(self.devices) < 2:
            self._ep_route = None
            return None
        # EP is off unless asked for, because the half of it that makes it correct is not
        # implemented. `MoE.forward` narrows `active` to the experts this rank owns
        # (`i % ep_world == ep_rank`) and relies on a driver that runs every rank and sums
        # the partials through `_ep_partials` -- and nothing in the tree creates that
        # bucket, nor can it: ExpertStore.fetch_async has no device argument, so a rank's
        # experts always land on the one card the layer lives on, and there is no way to
        # compute another card's share. Enabling it therefore computes half the experts
        # and silently drops the rest, which is worse than not using it: with EP off every
        # layer computes all six experts and the result is correct, it just costs the
        # duplicate weights. Turn it on only once a per-rank driver exists.
        if __import__("os").environ.get("TI_EP") != "1":
            self._ep_route = None
            return None
        try:
            from .ep import ExpertRoute

            n = int(getattr(self.cfg, "n_routed_experts", 0) or 0)
            if n <= 0:
                return None
            caps = [int(d.free) for d in self.devices]
            route = ExpertRoute(n, [f"cuda:{d.index}" for d in self.devices], caps)
            self._ep_route = route
            print(f"[tinyinfer] {route.summary()}", flush=True)
            print("[tinyinfer] WARNING: TI_EP=1 forces expert parallelism, but the per-rank "
                  "driver is not implemented -- each layer will compute only its own "
                  "rank's experts and the rest are dropped. Output WILL be wrong.",
                  flush=True)
            return route
        except Exception as exc:
            print(f"[tinyinfer] expert-parallel routing skipped ({exc!r})", flush=True)
            return None

    def _tensor_plan(self, name: str, out_features: int):
        """Where the output rows of one projection go, in the multi-GPU case.

        Returns [(device, (lo, hi)), ...] proportional to each card's free VRAM and
        aligned to the quantisation block, or None when there is only one card -- and
        None is what keeps the single-GPU code path exactly as it was.
        """
        if self.pool is None:
            return None
        try:
            from .tp import asplit

            caps = [self.pool.room_on(i) for i in range(len(self.pool.cap))]
            ranges = asplit(int(out_features), caps)
            if len(ranges) <= 1:
                return None
            return [(i, r) for i, r in enumerate(ranges)]
        except Exception:
            return None

    def _plan_experts(self) -> None:
        """Decide which card each expert belongs on.

        Capacity-proportional, not round-robin: on a 96+32 GB machine the big card takes
        three quarters of the experts, which is the point of asymmetric placement. With
        one card every expert maps to it, so the table is uniform and the lookup is a
        no-op -- the same code, not a special case.

        Per layer, because routing is per layer: expert 7 of layer 3 and expert 7 of layer
        4 are unrelated tensors and land wherever their own layer's plan puts them.
        """
        if self.placement is None:
            return
        cfg = getattr(self, "cfg", None)
        if cfg is None or self._expert_dev:
            return          # called before the config exists, or already planned
        n_layers = int(getattr(cfg, "n_layers", 0) or 0)
        per_layer = int(getattr(cfg, "n_routed_experts", 0) or 0)
        if not n_layers or not per_layer:
            return
        try:
            blob = int(getattr(cfg, "expert_bytes", 0) or 0)
        except Exception:
            blob = 0
        if blob <= 0:
            blob = 18_800_000          # measured flat-blob size on this checkpoint
        caps = [self.pool.room_on(i) for i in range(len(self.pool.cap))]
        # per layer: how many experts each card can hold at this blob size
        per = [max(0, c // blob) for c in caps]
        tot = sum(per)
        if tot <= 0:
            return
        for layer in range(n_layers):
            lo = 0
            for dev, cnt in enumerate(per):
                # this card's proportional share of THIS layer's experts
                share = int(round(per_layer * cnt / tot))
                take = min(per_layer - lo, share)
                for e in range(lo, lo + take):
                    self._expert_dev[(layer, e)] = dev
                lo += take
            # anything left over (rounding) goes to the biggest card
            if lo < per_layer:
                big = max(range(len(caps)), key=lambda i: caps[i])
                for e in range(lo, per_layer):
                    self._expert_dev[(layer, e)] = big
        counts = {}
        for d in self._expert_dev.values():
            counts[d] = counts.get(d, 0) + 1
        desc = ", ".join(f"cuda:{d}: {c} experts" for d, c in sorted(counts.items()))
        # Say what this actually is. _expert_dev is a static per-(layer, expert) card
        # assignment, and nothing in the decode path reads it: expert placement is done
        # dynamically by VramSlotCache, where any expert can land in any slot on any card.
        # Printing it as "the placement plan" made it look like the two agreed.
        _slots = getattr(getattr(self, "expert_store", None), "slots", None)
        _real = getattr(_slots, "slots", None)
        _note = (f" (static reference only; decode uses {_real} dynamic slots)"
                 if _real else "")
        print(f"[tinyinfer] expert placement plan: {desc}{_note}", flush=True)

    def expert_device(self, layer_id: int, expert_id: int) -> int:
        """Which card an expert is planned on (0 when single-GPU or unplanned)."""
        return getattr(self, "_expert_dev", {}).get((int(layer_id), int(expert_id)), 0)

    def block(self, layer_id: int) -> Block:
        if not self._expert_dev:
            self._plan_experts()   # config is live by the first call; plans once
        b = self._blocks.get(layer_id)
        if b is None:
            # The adapter decides how a layer is built. DeepSeek layers need the expert
            # store, quantisation, the compressor runtime and hyper-connection hooks;
            # Qwen layers need none of that. Keeping the branch here is what lets both
            # use the engine's streaming, residency and graph without a second engine.
            if getattr(self.adapter, "family", "") == "qwen3_5":
                # Tell the adapter how many materialised layers this machine can keep.
                # The plan already solved for the host budget; without this the adapter
                # would either hold every layer (47GB for the 27B, more than the box has)
                # or re-read each one from disk on every token.
                self._size_qwen_cache()
                b = self.adapter.build_block(layer_id, self.cfg)
                # Do NOT keep it in self._blocks. A DeepSeek block is a stub whose weights
                # are read per tensor from disk, so holding all 40 costs nothing; a Qwen
                # layer that has been materialised holds its real weights (731MB for the
                # 27B), and caching all 64 would need 47GB -- which is what filled the
                # machine's memory. The adapter owns Qwen's bounded cache instead.
                if layer_id < self.gpu_layers and self.device != "cpu":
                    b.to(self._gpu_for_layer(layer_id))
                return b
            b = Block(self.store, layer_id, self.cfg, expert_store=self.expert_store,
                      quant=self.tier.keep_weights_quantized, hashes=self.hashes,
                      # resident layers never leave the card, so their fp8 weight dequantisations are worth
                      # keeping (the per-token dequant of wo_a and of the shared expert is ~80ms
                      # across 40 layers)
                      cache_wo_a=(layer_id < self.gpu_layers and self.device != "cpu"),
                      cache_shared_bf16=(layer_id < self.gpu_layers and self.device != "cpu"),
                      expert_topk=self.tier.expert_topk, kv_quant=self._kv_quant,
                      rt=self._rt_for(layer_id), tp=getattr(self, "tp", None),
                      plan_fn=self._tensor_plan)
            b.hc_post_on_cpu = getattr(self, "small_ops_on_cpu", False)
            ffn = getattr(b, "ffn", None)
            if ffn is not None:
                # Expert parallelism, from the device list rather than a rank count: the
                # ranks are the cards this engine probed, so a one-card machine gets
                # world=1 and the filters below are no-ops. The legacy tp.rank path is
                # still honoured when a distributed group is actually initialised.
                _epr = self._expert_route()
                if _epr is not None:
                    ffn.ep_world = _epr.world
                    # Which rank this block is. Hardcoding 0 made every card serve rank
                    # 0's contiguous block and no card serve the rest: with world=2 the
                    # first 192 experts were computed twice and the last 192 never, and
                    # the answer looked plausible because a MoE layer with half its
                    # experts still produces well-scaled activations.
                    #
                    # The rank follows the device the block lives on. A block's device is
                    # known after placement, so resolve it now and fall back to 0 only when
                    # the device is not one of the route's (CPU-resident layer, or a
                    # single-card machine where world==1 and the value never matters).
                    _dev = getattr(b, "_device", None)
                    _rank = 0
                    if _dev is not None:
                        _ds = str(_dev)
                        if _ds in _epr.devices:
                            _rank = _epr.devices.index(_ds)
                    ffn.ep_rank = _rank
                    ffn._ep_route = _epr
                # DEFAULT OFF: the background fetch races the main-thread reads and made
                # the same MoE layer produce different values for different sequence
                # lengths (T=8 vs T=11 differed 4% on identical inputs), which showed up
                # as garbage tokens in the answer. Opt in with TI_ASYNC_FETCH=1.
                # tier.official_gemv: True forces the fused four-launch decode path,
                # False forces per-expert, None (the default) keeps the shape-based rule
                # that the field documented but never controlled.
                ffn._official_gemv = getattr(self.tier, "official_gemv", None)
                ffn._async_fetch = (bool(getattr(self.tier, "async_expert_fetch", False))
                                    and __import__("os").environ.get("TI_ASYNC_FETCH") == "1")
                ffn._batched_scatter = bool(getattr(self.tier, "batched_scatter", False))
                if self.tp is not None and self.tp.world > 1:
                    ffn.ep_world = self.tp.world
                    ffn.ep_rank = self.tp.rank
            if layer_id < self.gpu_layers and self.device != "cpu":
                b.to(self._gpu_for_layer(layer_id))
            self._blocks[layer_id] = b
        return b

    def _layer_device(self, b: Block) -> str:
        return getattr(b, "_device", "cpu")

    def _size_qwen_cache(self) -> None:
        if getattr(self, "_qwen_cache_sized", False):
            return
        self._qwen_cache_sized = True
        model = getattr(self.adapter, "model", None)
        if model is None or not hasattr(model, "set_ram_budget"):
            return
        # the model knows its per-layer size from the checkpoint
        n = int(getattr(model, "n_layers", 0) or 0)
        per_layer = 0.0
        try:
            prof = model.per_layer_bytes()
            per_layer = prof / (1 << 30)
        except Exception:
            per_layer = 0.0
        if per_layer <= 0 or n <= 0:
            return
        # the host budget the plan left for caching: everything not pinned by the
        # streaming ring and not needed for the KV/activation working set
        budget = float(getattr(self.hw_plan, "cache_bytes", 0) or 0) / (1 << 30)
        if budget <= 0:
            # no plan (CPU machine): keep the default the adapter was built with
            return
        model.set_ram_budget(budget, per_layer)

    def ensure_layer(self, layer_id: int):
        """Return the layer, positioned for this turn.

        Resident layers live on the card. Everything else is brought in for its turn and
        must be released afterwards (release_layer), so at most one streamed layer is on
        the device at a time: a 27B layer is 737MB of weights but ~2.4GB of VRAM once its
        slack and buffers are counted, and two of those do not fit in 6.9GB.
        """
        b = self.block(layer_id)
        if layer_id < self.gpu_layers:
            want = self._gpu_for_layer(layer_id)
            if getattr(b, "device", getattr(b, "_device", "cpu")) != want:
                b.to(want)
        return b

    def release_layer(self, layer_id: int) -> None:
        """Move a streamed layer back to the host so the next one can use the VRAM."""
        if layer_id < self.gpu_layers:
            return
        b = self._blocks.get(layer_id)
        if b is None:
            return
        cur = getattr(b, "device", getattr(b, "_device", "cpu"))
        if cur != "cpu":
            b.to("cpu")
            if self.device == "cuda":
                torch.cuda.empty_cache()

    def staticize_decode(self) -> int:
        """把 decode 路径切成「静态」：路由不再回 CPU，为 CUDA graph 捕获做准备。

        做法：用最近一次 forward 记录到的每层路由（MoE._last_ids）作为该层的固定
        专家集合。pin 住热点之后每层路由高度稳定，因此这个集合就是稳态下的真实
        路由 —— 而 decode 每层的路由本来也几乎是同一个集合。

        返回设置的层数。调用前必须先跑过 forward（让 _last_ids 有值）。
        """
        n = 0
        store = getattr(self, "expert_store", None)
        for _lid, b in self._blocks.items():
            ffn = getattr(b, "ffn", None)
            ids = getattr(ffn, "_last_ids", None) if ffn is not None else None
            if not ids:
                continue
            ffn._static_active = list(ids)
            ffn._static_decode = True
            # 预建 Expert 对象并预占槽位：捕获期间不能有 torch.empty / 新对象分配，
            # 否则 stream capture 直接被 invalidate。这些专家在稳态下本来就全在
            # 槽位里，这里只是把「第一次遇到」的分配挪到捕获之前。
            try:
                if store is not None:
                    recs = store.fetch(_lid, list(ids), cache=True)
                    for _e, _rec in zip(list(ids), recs):
                        if ffn._expert_cache.get(_e) is None:
                            ffn._expert_cache[_e] = ffn._make_expert(_rec)
            except Exception:
                pass
            n += 1
        if n:
            # 静态路径里不能再有每 token 的 .item()（那是捕获禁区）
            self.enable_prefetch = False
            self.predictor = None
        return n

    def pin_hot_experts(self, per_layer: int = 0) -> int:
        """把每层最常路由的专家固定进显存槽位（永不淘汰）。

        槽位是引擎的「手搓 manage」：1359 个 x 18.8MB 的固定区，谁在哪个槽位由
        VramSlotCache 管。pin 住热点之后，decode 的路由恒定落在槽位里，就不需要
        每层再 tolist() 回 CPU 决定该搬哪些专家 —— 那个 D2H 同步实测 3.3ms/层、
        每 token 130ms，是 decode 一半的时间。

        per_layer 默认 34：40 层 x 34 = 1360，正好是槽位规模。
        """
        slots = getattr(getattr(self, "expert_store", None), "slots", None)
        if slots is None or getattr(slots, "slots", 0) <= 0:
            return 0
        # 按「全局频率」而不是每层固定配额挑选：槽位总数就是预算。8GB 卡的 271 个
        # 和 L20 的 1359 个走同一套代码，规模由 slots 自己决定，不需要调用方知道。
        _cand = []
        for lid, b in self._blocks.items():
            counts = getattr(getattr(b, "ffn", None), "_route_counts", None)
            if counts:
                for _e, _c in counts.items():
                    _cand.append((_c, lid, _e))
        if not _cand:
            return 0
        _cand.sort(key=lambda t: -t[0])
        if per_layer:
            _seen: dict[int, int] = {}
            _sel = []
            for _c, _l, _e in _cand:
                if _seen.get(_l, 0) < per_layer:
                    _sel.append((_c, _l, _e))
                    _seen[_l] = _seen.get(_l, 0) + 1
            _cand = _sel
        # 槽位不能全 pin：剩下的要给「这一步新路由到的专家」腾地方。留一步的
        # 量（n_layers*topk）。池子连两步都装不下时（8GB 卡 271 < 480）pin 只会
        # 把池子锁死、decode 直接崩，这种规模就不 pin —— 让它在 LRU 上跑。
        _need_step = int(getattr(self.cfg, "n_layers", 40)) * int(
            getattr(self.cfg, "n_activated_experts", 6))
        _budget = int(slots.slots)
        if _budget < _need_step * 2:
            return 0
        # 覆盖率决定 pin 多少，而不是「排前 N 个」。
        # 实测（tests/test_hot_coverage.py，单层 384 次路由）：
        #   top-24 覆盖 56.5%、top-48 覆盖 74.7%、top-96 覆盖 92.4%。
        # 也就是说 pin 的收益是**次线性**的：pin 一半容量只换来一半覆盖率，
        # 而 pin 掉的槽位永久失效。真正划算的是「只 pin 覆盖率高得多的头部」。
        #
        # 这里按「pin 完仍至少留一个完整步的余量」定上限，再用覆盖率曲线检查：
        # 如果连 50% 路由都覆盖不到，pin 的收益低于它占掉的容量，直接不 pin。
        _avail = _budget - _need_step
        _tot = sum(c for c, _, _ in _cand)
        if _tot <= 0:
            return 0
        # 只 pin「贡献显著的头部」，不是「填满预算」。
        # 排序后的尾部每个只被路由寥寥几次：pin 它们既省不下一次搬运
        # （它们本来也几乎不会被路由到），又永久占死槽位。实测里 1119 个
        # 预算中会有 319 个落在「只被路由 1 次」的尾巴上。
        # 规则：累计覆盖率到 95% 为止，或到预算上限。
        _sel = []
        _acc = 0
        for _c, _l, _e in _cand[: _avail]:
            _sel.append((_c, _l, _e))
            _acc += _c
            if _acc >= 0.95 * _tot:
                break
        _cover = _acc / float(_tot)
        if _cover < 0.5:
            # 覆盖不到一半：pin 占掉的槽位比它省下的搬运更贵
            return 0
        _by_layer: dict[int, list[int]] = {}
        for _c, _l, _e in _sel:
            _by_layer.setdefault(_l, []).append(_e)
        pinned = 0
        for lid, es in _by_layer.items():
            # pin() 只固定「已在槽位里」的专家，而 prefill 收尾会清掉槽位，
            # 所以先把选中的搬进来，再固定。
            _miss = [e for e in es if not slots.has(lid, e)]
            if _miss:
                try:
                    self.expert_store.fetch(lid, _miss)
                except Exception:
                    pass
                for _e in _miss:
                    if not slots.has(lid, _e):
                        try:
                            slots.acquire(lid, _e)
                        except Exception:
                            pass
            for _expert in es:
                if slots.pin(lid, _expert):
                    pinned += 1
        # 把所有 pin 的结果一次性刷进 device 侧的槽位表（table_sync 内部是主机侧
        # 攒数组 + 一次 copy_，不是逐条 GPU 写）。这张表就是「GPU 侧调度」要用的
        # (layer, expert) -> slot 映射：kernel 用 index_select 直接拿槽位编号，
        # 不需要 CPU 每层 tolist 再发 H2D。
        if pinned:
            try:
                # fetch 的 DMA 是异步的：不先等它落地，表立刻指向这些槽位，decode
                # 读到的还是空槽位（实测 pin 后同一输入第二次 forward 出 NaN）。
                torch.cuda.synchronize()
                slots.slot_table(max(1, len(self._blocks)), int(getattr(
                    getattr(slots, "_index", None), "n_experts", 0) or 384))
                slots.table_sync()
                torch.cuda.synchronize()   # 表也必须落地，decode 下一步就会读它
                self._slot_tab_ready = True
                # 默认不设 _all_pinned：pin 只覆盖 prefill 期的 top-34，采样解码仍会
                # 路由到没 pin 的专家，那时表里是 -1，直通 kernel 就是非法地址，
                # miss 分支要继续兜底。TI_ALL_PINNED=1 强制走全命中路径，用来量
                # 「没有每层 D2H」时到底能跑多快。
                if __import__("os").environ.get("TI_ALL_PINNED") == "1":
                    for _b in self._blocks.values():
                        _ffn = getattr(_b, "ffn", None)
                        if _ffn is not None:
                            _ffn._all_pinned = True
            except Exception:
                self._slot_tab_ready = False
        # 槽位表就绪 = 可以捕获 decode graph 了。
        #
        # **但这里不立刻捕获。** 捕获需要捕获前先跑一次完整 forward 来建立
        # compress_kv / index_k（kv_source 层产出，后面 34 层直接读）；在 pin
        # 这一步跑的那次只有 1 个 token、位置 1，compressor 连一个完整分组都
        # 凑不齐，index_k 仍是 None，于是捕获时 compressor.py:285 直接抛
        #   TypeError: NoneType object is not subscriptable
        # 实测（L20）撞的就是这个。
        #
        # 真正的捕获交给第一次 decode：那时 prefill 已经跑过，_rt 里的状态是
        # 真实形状的。服务端 warmup() 之后紧接着就会有一次真实请求，代价只是
        # 第一个请求多花约 11 秒，换来正确性。
        self._graph_needed = bool(getattr(self.tier, "cuda_graph", False))
        return pinned

    def reset(self) -> None:
        self._last_routing.clear()
        # streaming completion events are keyed by layer and consumed by _flat_wait; a
        # stale entry makes the next forward rank a buffer copy that has not landed yet
        if hasattr(self, "_flat_ev"):
            self._flat_ev.clear()
        for _a in ("_flat_done", "_flat_slot_owner"):
            if hasattr(self, _a):
                getattr(self, _a).clear()
        if self.hashes is not None:
            self.hashes.reset()
        for b in self._blocks.values():
            if hasattr(b.attn, "reset"):
                b.attn.reset()
            # the compressor carries the tail of an incomplete group across chunks; if
            # it is not cleared a second run from position 0 starts with stale scores
            comp = getattr(b.attn, "compressor", None)
            if comp is not None and hasattr(comp, "reset"):
                comp.reset()
            idx = getattr(b.attn, "indexer", None)
            if idx is not None and hasattr(idx, "reset"):
                idx.reset()
            # the fp8 weight dequant is cached on the attention object; if the first
            # forward built it while a streamed layer's weights were still moving in,
            # every later run inherits those bytes
            if hasattr(b.attn, "_bf16_cache"):
                b.attn._bf16_cache = None
            # _expert_cache 必须清：里面的 Expert 持有的是槽位视图，而 slot 会在
            # evict 时被别的专家覆盖，留着就是悬空指针（试过不清，服务端每个请求
            # 都直接失败）。要跨请求复用，得让槽位本身不释放，而不是缓存对象。
            b.ffn._expert_cache.clear()
        # Streaming state is per-forward. Dropping it alone would leave every streamed
        # block's weights pointing into a ring buffer that no longer exists, so restore
        # the CPU references first and let the next forward rebuild the ring.
        # per-forward caches on the shared handoff slot must not leak between sessions
        try:
            from .deepseek.compressor import shared_attn as _sa0
            for _attr in ("_mask_cache", "candidates_by_chunk", "candidates",
                          "compress_kv", "topk_idxs"):
                if not hasattr(_sa0, _attr):
                    continue
                _v = getattr(_sa0, _attr)
                if _v is None:
                    continue
                if torch.is_tensor(_v):
                    # **不能把 tensor 设成 None**：它是压缩 KV 的存储，attention 的
                    # consumer 层直接读它。以前这里对 tensor 调 .clear() —— tensor
                    # 没有这个方法，AttributeError 被吞掉后走 setattr(..., None)，
                    # 于是 reset() 之后 compress_kv 变成 None，而 decode 路径**不会
                    # 重建它**：实测 prefill -> reset -> decode 得到全 NaN，且
                    # decode 之后 compress_kv 仍然是 None。
                    # 正确做法是清内容、保存储（槽位/形状不变，consumer 可直接读）。
                    try:
                        _v.zero_()
                    except Exception:
                        pass
                else:
                    try:
                        _v.clear()
                    except AttributeError:
                        setattr(_sa0, _attr, None)
        except Exception:
            pass
        # Streaming state is per-session. Re-point the streamed layers at their CPU
        # sources first (so no block holds a view into the ring), then release the pinned
        # host grant and drop the state: a rebuild that kept the old grant ran out of
        # budget and produced a partially initialised slab (silent numeric corruption).
        stream = self._stream_layers()
        for lid, b in self._blocks.items():
            if lid in stream:
                try:
                    b.to("cpu")
                except Exception:
                    pass
        # The slab/ring is a process-lifetime structure: rebuilding it per session leaks
        # the pinned grant and lets the caching allocator accumulate the old GPU buffers
        # (measured 17GB allocated on an 8GB card after three sessions). Keep it; only
        # the completion events belong to a single forward.
        if hasattr(self, "_flat_ev"):
            self._flat_ev.clear()
        if hasattr(self, "_flat_ev"):
            self._flat_ev.clear()
        if hasattr(self, "_flat_applies"):
            self._flat_applies = 0
        # inter-layer handoff slots: produced by a kv_source layer and consumed by the
        # layers after it inside one forward. A fresh run must not inherit the previous
        # run's compressed KV / index-K, which is what produced NaNs on a repeat prefill.
        try:
            from .deepseek.compressor import shared_attn as _sa
            _sa.reset()
        except Exception:
            pass

    # -- slot snapshots (multi-agent KV ownership) -------------------------
    def snapshot(self) -> dict:
        """Capture per-layer KV state so another slot can own the conversation."""
        snap = {}
        for i, b in self._blocks.items():
            a = b.attn
            entry = {"kv": None if a._kv_window is None else a._kv_window.clone(),
                     "klen": getattr(a, "_kv_len", 0)}
            ckv = getattr(a, "_compress_kv", None)
            entry["ckv"] = None if ckv is None else ckv.clone()
            comp = getattr(a, "compressor", None)
            if comp is not None:
                entry["pending"] = None if comp._pending_kv is None else (comp._pending_kv.clone(), comp._pending_score.clone())
            idx = getattr(a, "indexer", None)
            if idx is not None and getattr(idx, "_k_cache", None) is not None:
                entry["ik"] = idx._k_cache.clone()
            snap[i] = entry
        # The compressed layers hand state around through the shared handoff slot
        # (compress_kv / index_k / candidates are published by the kv_source layer and
        # read by its followers). Without these a restored session re-derives them on the
        # next forward and drifts: measured max|d| = 8.3 (rel 0.51) against a full
        # prefill while the argmax stayed the same.
        try:
            from .deepseek.compressor import shared_attn as _sa
            snap["_rt"] = {
                "compress_kv": None if getattr(_sa, "compress_kv", None) is None
                else _sa.compress_kv.clone(),
                "index_k": None if getattr(_sa, "index_k", None) is None
                else _sa.index_k.clone(),
                "topk_idxs": None if getattr(_sa, "topk_idxs", None) is None
                else _sa.topk_idxs.clone(),
                "candidates": None if getattr(_sa, "candidates", None) is None
                else _sa.candidates.clone(),
                "candidates_by_chunk": dict(getattr(_sa, "candidates_by_chunk", {}) or {}),
                "src_ratio": getattr(_sa, "src_ratio", None),
            }
        except Exception:
            snap["_rt"] = None
        return snap

    def restore(self, snap: dict) -> None:
        restore_rt = snap.pop("_rt", None) if isinstance(snap, dict) else None
        for i, entry in snap.items():
            if not isinstance(i, int):
                continue
            b = self.block(i)
            a = b.attn
            # **捕获期必须原地写回，不能 clone。**
            # restore() 在 _graph_try_capture 的 finally 里跑，而这时图已经把
            # 捕获期那些 buffer 的地址记进 kernel 参数了。clone() 会换一块新内存，
            # 于是图内 kernel 操作的地址和 Python 侧持有的地址从此分离：replay 时
            # 写的是旧地址（已被 allocator 回收），读回来就是垃圾 —— 实测 L1 的
            # 输出 absmax 是 2.36e37，正是 bf16 把别人的字节当数值解释的量级。
            # L0 侥幸正确是因为它的窗口路径不走这块 buffer。
            # 捕获刚结束（_graph_capturing 为 True）时保持原 buffer 不动：
            # 图已经把捕获期那些 buffer 的地址记进 kernel 参数了，clone 出来的
            # 新内存 fig 内根本看不见。
            # 只有「图已经真的建出来了」才 KEEP；warmup 那一次不算。
            # 原来判的是 _graph_just_captured，而它在 warmup 之前就置 True ——
            # 于是整个捕获流程里 KV 一次都没恢复，捕获那一轮看到的历史比 warmup
            # 多一个 token，路由漂到没进槽位的专家上，表里是 -1，kernel 拿
            # -1 * blob_bytes 当偏移读到别的专家：实测图内 L1 的
            # sids = [7, 6, 11, -1, -1, 13]，输出 absmax 2.36e37。
            # **图和 eager 必须共用同一块 KV buffer。**
            #
            # 以前这里给图单独留一份（_graph_bufs）再把 Python 侧换成 clone 出来的新
            # 显存，理由是「graph pool 的显存在图外读写是未定义行为」。那个理由本身
            # 没错，但分家的代价是致命的：图 replay 读写的是 A 块，prefill 走
            # _forward_impl 写的是 B 块，两套 cache 从此永不互通 —— 第一步 decode
            # 时图看到的 KV 里只有捕获那一步的内容，整个 prompt 的历史一个字节都
            # 没有。实测图解码退化成
            #   '<｜begin▁of▁sentence｜># 关于“ 关于“ 关于“新'
            # 而同一个 prompt 走 eager 是完整中文。
            #
            # 正确的解法是把「pool 里的显存不能在图外读写」这条约束在源头消掉：
            # 所有捕获期用到的 buffer 都在进入捕获窗口之前就分配好（见
            # _pregrow_graph_buffers），窗口内只复用、不分配，于是它们落在普通显存
            # 上，图外读写完全合法。这样图和 eager 天然是同一块地址。
            #
            # 捕获期确实写进去了一些中间 token 的 KV（warmup 那一步），必须清掉：
            # 用原地 copy_ 把捕获前那份快照写回去，地址不变。
            if entry.get("kv") is None:
                _kv = None
            else:
                _snapkv = entry["kv"]
                _cur = a._kv_window
                if (_cur is not None and _cur.shape == _snapkv.shape
                        and _cur.dtype == _snapkv.dtype and _cur.device == _snapkv.device):
                    # 同一块、原地回填：地址保持不变，图里记的就是它。
                    _cur.copy_(_snapkv)
                    _kv = _cur
                else:
                    _kv = _snapkv.clone()
            # 环形槽位 cache 之后 _kv_window 和 _kv_buf 是同一块 buffer（forward 里
            # 直接 self._kv_window = buf），两个 property 各自映射到自己的 state key
            # （attention 是 "buf"，compressed 是 "winbuf"），所以两处都要写 ——
            # 只写 _kv_window 的话 forward 会从 _kv_buf 现开一块空 buffer，
            # 恢复出来的历史一个字节都用不上。
            if _kv != "KEEP":
                a._kv_window = _kv
                a._kv_buf = _kv
            if hasattr(a, "_kv_len"):
                a._kv_len = int(entry.get("klen", 0))
            if hasattr(a, "_compress_kv"):
                a._compress_kv = None if entry.get("ckv") is None else entry["ckv"].clone()
            comp = getattr(a, "compressor", None)
            if comp is not None:
                pending = entry.get("pending")
                if pending is None:
                    comp._pending_kv = comp._pending_score = None
                else:
                    comp._pending_kv, comp._pending_score = pending[0].clone(), pending[1].clone()
            idx = getattr(a, "indexer", None)
            if idx is not None:
                idx._k_cache = None if entry.get("ik") is None else entry["ik"].clone()
            b.ffn._expert_cache.clear()
        if restore_rt:
            try:
                from .deepseek.compressor import shared_attn as _sa
                for _k in ("compress_kv", "index_k", "topk_idxs", "candidates"):
                    _v = restore_rt.get(_k)
                    setattr(_sa, _k, None if _v is None else _v.clone())
                _sa.candidates_by_chunk = dict(restore_rt.get("candidates_by_chunk") or {})
                _sa.src_ratio = restore_rt.get("src_ratio")
            except Exception:
                pass

    # -- persistent KV cache (tinyinfer/.kvcache) ---------------------------
    @property
    def kv_cache_dir(self):
        from pathlib import Path as _P
        return _P(__file__).resolve().parent.parent.parent / ".kvcache"

    def save_kv_cache(self, name: str):
        """Persist this session's KV/compressed state under .kvcache/<name>.pt.

        The file holds the per-layer KV window, compressed-KV handoff slot and indexer
        caches (see snapshot()), so a later process can resume the same conversation
        without re-prefilling it.
        """
        import torch as _t
        path = self.kv_cache_dir / (str(name) + ".pt")
        path.parent.mkdir(parents=True, exist_ok=True)
        _t.save(self.snapshot(), path)
        return path

    def load_kv_cache(self, name: str) -> bool:
        """Restore a previously saved session state; False when the file is absent."""
        import torch as _t
        path = self.kv_cache_dir / (str(name) + ".pt")
        if not path.exists():
            return False
        self.restore(_t.load(path, map_location="cpu", weights_only=False))
        return True

    # -- double-buffered streaming -----------------------------------------
    def _ensure_stream_buffers(self, layer_id: int):
        """Two flat device buffers per EXACT weight signature (v5).

        v4 grouped by (ratio>0, compressor, indexer) and broke at the L20 boundary
        (ratio=1 kv_source has a different weight count than the ratio=1 consumers).
        v5 keys the cache by the full (name, shape) signature, so layers that can
        share a flat buffer are exactly the ones whose weight list is identical.
        """
        b = self.block(layer_id)
        paths = b._weight_paths()
        sig = tuple((name, tuple(t.shape)) for (_, name, t) in paths)
        cache = getattr(self, "_stream_buffer_cache", None)
        if cache is None:
            self._stream_buffer_cache = cache = {}
        ent = cache.get(sig)
        if ent is None and len(cache) >= 1:
            # VRAM guard: one signature group x 2 buffers (2 x 190MB = 380MB); later
            # groups fall back to the synchronous per-layer copy.
            return None, None
        if ent is None:
            layout = []
            off = 0
            for owner, name, t in paths:
                n = t.numel() * t.element_size()
                off = (off + 255) // 256 * 256
                layout.append((owner, name, None, off, n))
                off += n
            bufs = [torch.empty(off, dtype=torch.uint8, device=self.device) for _ in range(2)]
            ent = cache[sig] = (layout, bufs)
        return ent

    def _stream_copy_sync(self, layer_id: int) -> None:
        """Copy a streamed layer's weights into its rotating buffer on the compute stream."""
        (layout, bufs) = self._ensure_stream_buffers(layer_id)
        b = self.block(layer_id)
        paths = b._weight_paths()
        buf = bufs[layer_id % 2]
        for (owner, name, t), (_, _, _, off, _) in zip(paths, layout):
            buf[off:off + t.numel() * t.element_size()].view(t.dtype).view(t.shape).copy_(t)
            set_weight(owner, name, buf[off:off + t.numel() * t.element_size()].view(t.dtype).view(t.shape))
        if getattr(b, "engram", None) is not None:
            b.engram.to(self.device)
        b._device = self.device

    # -- flat streaming (v6) -----------------------------------------------
    # Every streamed layer reuses ONE pair of device buffers sized to the largest
    # layer, and every layer's host weights are packed once into a pinned flat
    # buffer. A decode step then issues one 442MB pinned H2D per layer (16ms at
    # pinned bandwidth) instead of ~31 pageable per-tensor copies that each cost a
    # host round-trip; the next layer is copied on a side stream while the current
    # one computes, so the transfer hides behind the GPU.
    def _flat_enabled(self) -> bool:
        st = getattr(self, "_flat_state", None)
        if st == "off":
            return False
        if st is None:
            import os as _os

            flag = _os.environ.get("TI_FLAT")
            if flag is not None and flag != "1":
                self._flat_state = "off"
                return False
            if flag is None and self.gpu_layers >= self.max_layers:
                # every layer is resident: there is nothing to stream
                self._flat_state = "off"
                return False
            try:
                self._flat_setup()
            except Exception as exc:  # pinned-pool exhaustion, OOM: stay on the old path
                print(f"[tinyinfer] flat streaming off: {exc!r}", flush=True)
                self._flat_state = "off"
                return False
        return True

    def _warm_weights(self, layers) -> None:
        """Read every streamed layer's weights through an 8-way pool first.

        ShardStore.get is one open+seek+read per tensor (31 per layer), which measured
        ~1.2GB/s on this box; the same bytes through 8 threads saturate far better and
        leave the block constructors reading from the OS cache.
        """
        import itertools
        from concurrent.futures import ThreadPoolExecutor

        names: list[str] = []
        try:
            wm = self.store.weight_map
        except AttributeError:
            return
        for i in layers:
            pre = f"layers.{i}."
            exp = pre + "ffn.experts."
            # non-expert weights only: the routed experts are 12.75GB/layer and are
            # streamed per token, warming them here would just OOM the box
            names.extend(n for n in wm if n.startswith(pre) and not n.startswith(exp))
        if not names:
            return
        import time as _time
        t0 = _time.time()
        total = 0
        pool = ThreadPoolExecutor(max_workers=8)
        it = iter(names)
        try:
            while True:
                batch = list(itertools.islice(it, 8))
                if not batch:
                    break
                for t in pool.map(self.store.get, batch):
                    total += t.numel() * t.element_size()
        finally:
            pool.shutdown(wait=True)
        dt = _time.time() - t0
        print(f"[tinyinfer] prewarm {total/1e9:.1f}GB of layer weights in {dt:.1f}s "
              f"= {total/1e9/max(dt,1e-6):.1f}GB/s", flush=True)

    def _flat_setup(self) -> None:
        import os

        streamed = sorted(self._stream_layers())
        if not streamed:
            raise RuntimeError("no streamed layers")
        lay: dict[int, tuple[list[tuple[int, int]], int]] = {}
        sizes: dict[int, int] = {}
        # Layers already lying flat in the pool. Weights are read straight into pool
        # slices, so a layer's tensors are often one contiguous region already; those
        # layers can be streamed from their own bytes and never get a second copy.
        zero_copy: set[int] = set()
        _pool = getattr(self, "host_pool", None)
        _pool_base = getattr(_pool, "_base", None) if _pool is not None else None
        for i in streamed:
            paths = self.block(i)._weight_paths()
            # Only layers whose weights ALL live in the host pool can be streamed from
            # their own bytes: wqkv.weight is a device copy that stays resident, and its
            # data_ptr() is a GPU address, which would make the span arithmetic nonsense.
            if _pool_base is not None and paths and all(
                    t.device.type == "cpu" for (_, _, t) in paths):
                try:
                    actual = [(int(t.data_ptr()) - _pool_base,
                               t.numel() * t.element_size()) for (_, _, t) in paths]
                except Exception:
                    actual = None
                if actual and all(o >= 0 for o, _ in actual):
                    pos = sorted((o, o + n) for o, n in actual)
                    no_overlap = all(pos[k][1] <= pos[k + 1][0]
                                     for k in range(len(pos) - 1))
                    span = max(e for _, e in pos)
                    total_b = sum(n for _, n in actual)
                    # 4096-byte allocation alignment leaves padding; bound it so a
                    # pathological layout falls back to the packed copy instead
                    if __import__("os").environ.get("TI_FLAT_DBG"):
                        print(f"[zc] L{i} base={_pool_base} n={len(actual)} span={span} "
                              f"data={total_b} slack={span - total_b} "
                              f"no_overlap={no_overlap}", flush=True)
                    # The zero-copy path is opt-in: cudaHostAlloc's host address and the address
                    # torch.frombuffer reports for the same slice do not agree on Windows
                    # (measured: base 8.67e9 vs data_ptr 5.3e12), so the span arithmetic
                    # cannot be trusted until the loader lays weights out for it directly.
                    # See docs/模型结构.md; TI_ZC=1 turns the experiment on.
                    if (no_overlap and span <= total_b + 65536 * len(actual)
                            and __import__("os").environ.get("TI_ZC") == "1"):
                        lay[i] = (actual, span)
                        sizes[i] = span
                        zero_copy.add(i)
                        continue
            offs: list[tuple[int, int]] = []
            off = 0
            for (_, _, t) in paths:
                n = t.numel() * t.element_size()
                off = (off + 255) // 256 * 256
                offs.append((off, n))
                off += n
            lay[i] = (offs, off)
            sizes[i] = off
        # No second host copy: Block.to() already keeps a pinned ref for every weight
        # (_cpu_refs, 15.5GB on this checkpoint). A packed flat copy on top of that is
        # another 15.5GB, which pushed the box into paging (rss 21.6GB vs 14.6GB
        # available). Instead the per-tensor pinned copies go straight into the shared
        # pair: still async, still one H2D per layer, no extra host memory.
        keep: list[int] = []
        mx = max(sizes.values())
        # Three rotating buffers. With two, the H2D for layer k+1 starts only when
        # layer k begins, and a 719MB transfer (27ms) is slower than one layer of
        # compute (~15ms), so it can never hide. Three let layer k compute while
        # k+1 and k+2 stream in.
        # ring depth is measurable: 3 hides one H2D behind compute, 2 frees a layer-sized
        # buffer (463MB here) for one more resident layer on a card this tight.
        # Re-measured after the fp4 experts moved to vLLM's Marlin (the experts got 4.8x
        # faster, so the layer H2D is now a larger share of the step): depth 3 =
        # 0.935-1.19s/token = 0.968 tok/s, depth 4 = 1.03-1.40s/token = 0.864,
        # depth 2 = 1.04-1.27s/token = 0.867. Deeper is not free -- each extra ring slot
        # costs 463MB of the 6.9GB card -- so 3 is the measured optimum, not a maximum.
        _ring_n = max(2, min(4, int(__import__("os").environ.get("TI_RING", "3"))))
        bufs = [torch.empty(mx, dtype=torch.uint8, device=self.device) for _ in range(_ring_n)]
        pinned_gb = 0.0
        # Pinned budget. Pinning all 17.9GB of streamed weights starved the expert
        # cache: pinned sources are the only ones that can use async DMA, and the
        # pinned pool also has to hold the expert blobs.
        from .budget import budget as _host_budget

        _hb = _host_budget()
        # Per-tensor fallback only: pin as much as the budget has left, one tensor at
        # a time, and let the ones that do not fit stay pageable.
        pin_budget = int(float(__import__("os").environ.get("TI_FLAT_PIN_GB", "0")) * (1 << 30))
        if pin_budget <= 0:
            pin_budget = min(_hb.room(), int(_hb.total * 0.36))
        # One pinned slab for every streamed layer, laid out exactly like the device
        # buffers, so a layer costs ONE flat H2D instead of 15-30 small copy_ calls.
        # cProfile: the per-tensor path issued 1469 copy_ calls and burned 1.01s of a
        # 3.45s decode step in host-side dispatch, while the pinned bytes were already
        # resident (the slab replaces them, it does not add to them).
        slab_total = sum(sizes[i] for i in streamed) + 512
        # Pinning every streamed layer is not the goal; it is only worth what it saves.
        # On a box where the weights alone are half of RAM, pin at most half of the
        # remaining room so the page cache (and the experts) still have memory.
        # Cap the pinned set at a share of PHYSICAL RAM, not of the volatile "room":
        # the slab plus the expert cache plus the 17GB of streamed weights must leave
        # the page cache alive, or the SSD drops from 4.6 to 3.1GB/s and every fetch
        # doubles (measured: full-RAM slab -> 2.29s fetch, capped slab -> 1.47s).
        # Fixed share of PHYSICAL RAM, not of the volatile room: the room shrinks as the
        # model weights load, so a room-derived cap collapsed to 6.7GB and left 21 of 35
        # layers on the slow per-tensor path. The invariant that matters is
        # weights(17GB) + slab + expert cache <= RAM - OS, and the planner already sized
        # the cache from the same budget, so the slab takes its share of the total.
        # The slab and the expert cache are carved from the same pinned pool, and at
        # decode it is the cache that pays: a larger slab means more streamed layers get
        # their weights pre-staged, while a larger cache means fewer 18.8MB expert reads
        # per token. Measured steady-state decode (0.66 -> 0.75 hit rate takes the step
        # from 574ms to 519ms), so the cache is worth more than the slab. TI_SLAB_FRAC
        # lets the split be measured instead of argued.
        _slab_frac = float(__import__("os").environ.get("TI_SLAB_FRAC", "0.28"))
        _slab_cap = int(_hb.total * _slab_frac)
        if slab_total > _slab_cap:
            slab_total = _slab_cap
        if __import__("os").environ.get("TI_FLAT_DBG"):
            _f = sizes[streamed[0]] if streamed else 0
            print(f"[slabdbg] n={len(streamed)} slab_total={slab_total/2**30:.2f}GB "
                  f"cap={_slab_cap/2**30:.2f}GB hb_total={_hb.total/2**30:.1f}GB "
                  f"room={_hb.room()/2**30:.1f}GB first={_f/2**20:.0f}MB", flush=True)
        host_slab = None
        # Ask the host budget, which re-reads what the machine actually has left right
        # now. No flag, no per-machine constant: on a small box the grant comes back
        # short and we fall back to per-tensor pinned copies, on a big one we get the
        # whole slab. The slab REPLACES those copies, so the freed memory is available
        # to the next request automatically (available() is re-read every time).
        # One pinned block PER LAYER rather than one giant block: a single 15GB
        # cudaHostAlloc is refused by the driver on some Windows configurations (the
        # per-tensor fallback below pinned the same 14.9GB in small pieces), while a
        # 468MB block always succeeds. Each block is still laid out exactly like the
        # device buffer, so a layer is one flat H2D, which is the whole point.
        host_map: dict[int, torch.Tensor] = {}
        _why = "ok"
        host_slab: dict[int, torch.Tensor] | None = None
        self._host_slab = None
        # One layer at a time: allocate its block, copy the weights in, point the layer
        # at the block, then let the old pageable copy go. Allocating all blocks first
        # holds both copies at once (+18GB) and that is what pushed this box into swap.
        _slab_used = 0
        _pool = getattr(self, "host_pool", None)
        if _pool is not None:
            # The slab is built from pool slices (build_pool put the weights there), so
            # both the source and the slab live inside the claimed pool; nothing here
            # touches private host memory.
            if __import__("os").environ.get("TI_FLAT_DBG"):
                print(f"[flatdbg] pool total={_pool.total / 2**30:.2f}GB used="
                      f"{_pool.used / 2**30:.2f}GB free={_pool.free / 2**30:.2f}GB "
                      f"slab_total={slab_total / 2**30:.2f}GB", flush=True)
            slab_total = min(slab_total, _pool.free)
        if slab_total <= 0 and _pool is not None:
            _why = "pool-backed"
        for i in streamed:
            need = sizes[i]
            if i in zero_copy and _pool is not None:
                # Already flat in the pool: reuse the bytes in place. Copying them into
                # a slab block is what made the slab run out of room (5/30 layers) and
                # dropped the disk to 1.4GB/s.
                b = self.block(i)
                base_off = min(o for o, _ in lay[i][0])
                mv = _pool.slice_at(base_off, need)
                if mv is not None:
                    blk = torch.frombuffer(mv, dtype=torch.uint8)
                    blk._tiny_pool_mv = mv
                    refs = {}
                    for (owner, name, t), (o, n) in zip(b._weight_paths(), lay[i][0]):
                        refs[Block._wkey(owner, name)] = blk[o:o + n].view(t.dtype).view(t.shape)
                    b._cpu_refs = refs
                    b._device = "cpu"
                    host_map[i] = blk
                    keep.append(i)
                    continue
            if _slab_used + need > slab_total:
                if _pool is None:
                    _why = (f"reached the pinned cap {slab_total/2**30:.1f}GB "
                            f"({len(host_map)}/{len(streamed)} layers)")
                break
            if _pool is not None:
                blk = _pool.tensor(need)
                if blk is None:
                    _why = (f"pool exhausted at layer {i} "
                            f"({_pool.used/2**30:.1f}/{_pool.total/2**30:.1f}GB used)")
                    break
            else:
                got = _hb.grant(need)
                if got < need:
                    _why = f"budget: wanted {need/2**20:.0f}MB got {got/2**20:.0f}MB ({_hb.stats()})"
                    break
                try:
                    blk = torch.empty(need, dtype=torch.uint8, pin_memory=True)
                except Exception as exc:
                    _hb.release(need)
                    _why = f"pinned alloc failed at layer {i}: {exc!r}"
                    break
            _slab_used += need
            self._slab_granted = getattr(self, "_slab_granted", 0) + need
            b = self.block(i)
            paths = b._weight_paths()
            offs = lay[i][0]
            for (owner, name, t), (o, n) in zip(paths, offs):
                dst = blk[o:o + n].view(t.dtype).view(t.shape)
                try:
                    dst.copy_(t)
                except Exception:
                    pass
                set_weight(owner, name, dst)
            _refs = {}
            for (owner, name, t), (o, n) in zip(paths, offs):
                _refs[Block._wkey(owner, name)] = blk[o:o + n].view(t.dtype).view(t.shape)
            b._cpu_refs = _refs
            b._device = "cpu"
            host_map[i] = blk
            del paths, offs
            keep.append(i)
        if len(host_map) == len(streamed):
            pinned_gb = sum(int(v.numel()) for v in host_map.values()) / 2 ** 30
            self._flat_state = (lay, bufs, host_map)
            self._host_slab = host_map
        elif host_map:
            # Partial slab: the layers that got a block keep the one-copy fast path, the
            # rest fall back to per-tensor pinned copies. All-or-nothing wasted the RAM
            # that WAS available and left every layer on the slow path.
            print(f"[tinyinfer] partial slab {len(host_map)}/{len(streamed)} layers ({_why})",
                  flush=True)
            pinned_gb = sum(int(v.numel()) for v in host_map.values()) / 2 ** 30
            # the slab already consumed its share of the pinned budget; counting from
            # zero here pinned a second full budget and took the box to its page file
            pinned_used = int(pinned_gb * 2 ** 30)
            for i in streamed:
                if i in host_map:
                    continue
                b = self.block(i)
                paths = b._weight_paths()
                refs = []
                for (_, _, t) in paths:
                    nbytes = t.numel() * t.element_size()
                    if (t.is_cpu and not t.is_pinned() and pinned_used + nbytes <= pin_budget):
                        try:
                            t = t.pin_memory()
                            pinned_used += nbytes
                        except Exception:
                            pass
                    refs.append(t)
                # keyed the same way Block.to() keys its map, so the two stay in sync
                b._cpu_refs = {Block._wkey(o, n): t for (o, n, _), t in zip(paths, refs)}
                for (owner, name, _), t in zip(paths, refs):
                    set_weight(owner, name, t)
                b._device = "cpu"
                pinned_gb += sum(t.numel() * t.element_size() for t in refs if t.is_pinned()) / 2 ** 30
                keep.append(i)
            self._flat_state = (lay, bufs, host_map)
            self._host_slab = host_map
        else:
            print(f"[tinyinfer] slab unavailable ({_why}); falling back to per-tensor pinned",
                  flush=True)
            pinned_used = 0
            for i in streamed:
                b = self.block(i)
                paths = b._weight_paths()
                refs = []
                for (_, _, t) in paths:
                    nbytes = t.numel() * t.element_size()
                    if (t.is_cpu and not t.is_pinned() and pinned_used + nbytes <= pin_budget):
                        try:
                            t = t.pin_memory()
                            pinned_used += nbytes
                        except Exception:
                            pass
                    refs.append(t)
                b._cpu_refs = {Block._wkey(o, n): t for (o, n, _), t in zip(paths, refs)}
                # repoint the layer at the pinned copies so the pageable originals can be
                # freed: keeping both meant 15.9GB + 15.9GB resident, and under that pressure
                # the pinned H2D fell from 30GB/s to 6.5GB/s (measured T=1 vs T=2)
                for (owner, name, _), t in zip(paths, refs):
                    set_weight(owner, name, t)
                b._device = "cpu"
                pinned_gb += sum(t.numel() * t.element_size() for t in refs if t.is_pinned()) / 2 ** 30
                keep.append(i)
            self._flat_state = (lay, bufs, None)
        self._flat_ev = {}
        self._copy_stream = torch.cuda.Stream(device=self.device)
        print(f"[tinyinfer] flat streaming: {len(keep)} layers, {mx/2**20:.0f}MBx3 VRAM, "
              f"{pinned_gb:.1f}GB {'slab' if self._host_slab else 'per-tensor'} pinned source",
              flush=True)

    def _flat_apply(self, layer: int, buf_idx: int) -> None:
        self._flat_applies = getattr(self, "_flat_applies", 0) + 1
        lay, bufs, host = self._flat_state
        b = self.block(layer)
        offs, total = lay[layer]
        buf = bufs[buf_idx]
        if __import__("os").environ.get("TI_DEBUG_FLAT"):
            print(f"[flat] layer {layer} buf {buf_idx} slab={host is not None and layer in host}",
                  flush=True)
        if host is None or layer not in host:
            # Per-tensor copies from the pinned refs Block already holds (a partial slab
            # leaves most layers on this path). The refs ARE pinned host memory, so issue
            # the DMA straight at it: tensor.copy_() costs ~84us of CPU dispatch per call
            # (measured in the decode trace: 2849 calls per step, 239ms of the step) and
            # the step is CPU bound -- the GPU kernels are only ~91ms of it.
            from .loading import _async_h2d

            # 逐 tensor 提交 DMA：每层几十个，每次 copy_ 约 84us 的 CPU dispatch，
            # 全局每步 2849 次调用 -> 239ms，而 GPU kernel 只有 ~91ms —— 拷贝提交
            # 本身就是瓶颈。这里把手上的 (dst, src, n) 一次收齐再提交，让驱动按批
            # 入队而不是每个都走一遍 Python + dispatch。
            _pairs = []
            for (owner, name, t), (o, n) in zip(b._weight_paths(), offs):
                src = b._cpu_refs[Block._wkey(owner, name)]
                dst = buf[o:o + n].view(t.dtype).view(t.shape)
                _pairs.append((dst, src, n))
            _queued = False
            if _pairs:
                _batch = getattr(b, "_h2d_batch", None)
                if _batch is not None:
                    _queued = _batch(_pairs)
            if not _queued:
                for dst, src, n in _pairs:
                    if not _async_h2d(dst, src, n):
                        dst.copy_(src, non_blocking=True)
        else:
            buf[:total].copy_(host[layer], non_blocking=True)
        for (owner, name, t), (o, n) in zip(b._weight_paths(), offs):
            set_weight(owner, name, buf[o:o + n].view(t.dtype).view(t.shape))
        if getattr(b, "engram", None) is not None:
            b.engram.to(self.device)
        b._device = self.device
    def _flat_slot_guard(self, layer: int, buf_idx: int) -> None:
        """Stop the copy stream from refilling a ring slot the GPU may still be reading.

        With a three-deep ring and two layers prefetched ahead, layer i+3 lands in the
        same slot as layer i. The copy stream is independent of the compute stream, so
        without this the DMA could overwrite layer i's weights while its kernels were
        still running -- and since the reused memory held a *previous* run's bytes, the
        damage decayed instead of failing outright: four back-to-back runs of the same
        prompt produced '<eos>', then gibberish, then nearly right, then the correct
        answer.
        """
        if getattr(self, "_flat_slot_owner", None) is None:
            self._flat_slot_owner = {}
            self._flat_done = {}
        prev = self._flat_slot_owner.get(buf_idx)
        if prev is not None and prev != layer:
            done = self._flat_done.pop(prev, None)
            if done is not None:
                self._copy_stream.wait_event(done)
        self._flat_slot_owner[buf_idx] = layer

    def _flat_prefetch(self, layer: int, buf_idx: int) -> None:
        if layer not in self._flat_state[0]:
            return
        self._flat_slot_guard(layer, buf_idx)
        ev = torch.cuda.Event()
        with torch.cuda.stream(self._copy_stream):
            self._flat_apply(layer, buf_idx)
            ev.record(self._copy_stream)
        self._flat_ev[layer] = ev

    def _flat_wait(self, layer: int, buf_idx: int) -> None:
        import time as _t
        t0 = _t.perf_counter()
        try:
            if layer not in self._flat_state[0]:
                self.block(layer).to(self.device)
                return
            ev = self._flat_ev.pop(layer, None)
            if ev is not None:
                torch.cuda.current_stream().wait_event(ev)
            else:
                self._flat_slot_guard(layer, buf_idx)
                self._flat_apply(layer, buf_idx)
        finally:
            self._h2d_wait_ms = getattr(self, "_h2d_wait_ms", 0.0) + (_t.perf_counter() - t0) * 1e3

    def _stream_layers(self):
        """Layers that are re-copied every token (the tail beyond the resident set)."""
        if getattr(self, "_stream_set", None) is None:
            if self.layer_ids is not None:
                # gpu_layers 是 planner 按显存算出来的「能常驻几层」，而 layer_ids
                # 是调用方实际要建的层（8GB 卡上常常只有 4 层）。两者不一致时，
                # 直接用 gpu_layers 当切片下标是错的：layer_ids=[0..3] 配
                # gpu_layers=11 会切出空集（恰好对），但配 gpu_layers=2 就会
                # 把 [2,3] 当成 streamed —— 而它们其实常驻。
                # 正确语义：常驻的是 layer_ids 的前 min(gpu_layers, len) 个。
                _n_res = min(int(self.gpu_layers), len(self.layer_ids))
                self._stream_set = set(self.layer_ids[_n_res:])
            else:
                self._stream_set = set(range(min(int(self.gpu_layers), self.max_layers),
                                             self.max_layers))
        return self._stream_set

    def _stream_prefetch(self, layer_id: int, stream):
        """Async-copy a streamed layer's weights into its rotating buffer (copy stream).

        v5: the buffer is keyed by the exact weight signature, so layout and the
        current layer always agree (L20 has its own buffers). Pinned CPU refs make
        the H2D run at pinned bandwidth. Returns the CUDA event to wait on before
        the layer computes.
        """
        (layout, bufs) = self._ensure_stream_buffers(layer_id)
        b = self.block(layer_id)
        paths = b._weight_paths()
        if b._cpu_refs is None:
            b._cpu_refs = [t.pin_memory() if t.is_cpu and t.device.type == "cpu" else t
                           for (_, _, t) in paths]
        if len(paths) != len(layout):
            raise RuntimeError(f"L{layer_id}: {len(paths)} weights vs layout {len(layout)}")
        buf = bufs[layer_id % 2]
        ev = torch.cuda.Event()
        with torch.cuda.stream(stream):
            for (owner, name, t), cpu, (_, _, _, off, n) in zip(paths, b._cpu_refs, layout):
                buf[off:off + n].view(t.dtype).view(t.shape).copy_(cpu, non_blocking=True)
            ev.record(stream)
        for (owner, name, t), (_, _, _, off, n) in zip(paths, layout):
            set_weight(owner, name, buf[off:off + n].view(t.dtype).view(t.shape))
        if getattr(b, "engram", None) is not None:
            b.engram.to(self.device)
        return ev

    # -- streaming slots ---------------------------------------------------
    def _ensure_slots(self):
        if getattr(self, "_stream_slots", None) is None:
            self._stream_slots = {}
        return self._stream_slots

    def _slot_for(self, layer_id: int):
        """Two rotating device slots per attention kind; a streamed layer copies its
        weights into one, so the H2D is reused instead of allocating per token."""
        slots = self._ensure_slots()
        b = self.block(layer_id)
        # a slot must match the exact weight layout: compress ratios plus whether the
        # layer owns a compressor / indexer (source layers carry them, consumers don't)
        kind = (self.cfg.compress_ratios[layer_id] > 0,
                getattr(b.attn, "compressor", None) is not None,
                getattr(b.attn, "indexer", None) is not None)
        pair = slots.get(kind)
        if pair is None:
            pair = slots[kind] = []
        while len(pair) < 2:
            blk = Block(self.store, layer_id, self.cfg, expert_store=self.expert_store,
                        quant=self.tier.keep_weights_quantized, hashes=None,
                        expert_topk=self.tier.expert_topk, kv_quant=self._kv_quant,
                        rt=self._rt_for(layer_id))
            blk.hc_post_on_cpu = getattr(self, "small_ops_on_cpu", False)
            blk.to(self.device)
            pair.append(blk)
        return pair[layer_id % 2]

    def _stream_into_slot(self, block_i: Block, slot: Block) -> None:
        if block_i._cpu_refs is None:
            block_i._cpu_refs = [t.pin_memory() if t.is_cpu and t.device.type == "cpu" else t
                                 for (_, _, t) in block_i._weight_paths()]
        dst = slot._weight_paths()
        for (owner, name, t), cpu in zip(dst, block_i._cpu_refs):
            t.copy_(cpu, non_blocking=True)

    def _prefetch_layer(self, layer_id: int, experts) -> None:
        """Stage the experts for a later layer: one disk read, straight into a VRAM slot.

        The old version called both the RAM prefetch and the VRAM prefetch, so every
        blob was read from disk twice and the slot pool (measured 0 hits) was refilled
        with data that was never consumed.
        """
        b = self.block(layer_id)
        # RAM only. Staging into VRAM here filled the 24-slot pool with 186 predicted
        # blobs per token and evicted the very ones the compute thread was about to
        # use (measured: 2.4k loads, 1 hit). Slots are now H2D staging for the layer
        # being executed, and this thread just warms the RAM cache ahead of it.
        b.ffn.prefetch(experts)

    def _warm_next_layer(self, layer_id: int, experts) -> int:
        """把下一层要用的专家**搬进槽位**（不是只预热 RAM）。

        这是整条 decode 流水线能不能藏住 IO 的关键。当前的做法是「算到哪一层
        才去搬那一层的专家」——MoE 一开始就得等 6 个专家到齐，40 层叠加起来
        就是每步 800ms 的等待（实测：一层 80ms，其中 78ms 是等专家）。

        改成在前一层的 attention 还在算的时候就把下一层的专家发出去：搬运和
        计算重叠，等真正进到那一层的 MoE 时数据已经到了。

        返回真正发起的搬运个数。
        """
        if not experts:
            return 0
        b = self.block(layer_id)
        ffn = getattr(b, "ffn", None)
        if ffn is None:
            return 0
        st = getattr(ffn, "_expert_store", None)
        if st is None or getattr(st, "slots", None) is None:
            return 0
        if int(getattr(st.slots, "slots", 0) or 0) <= 0:
            return 0
        try:
            mgr = ffn._tier_manager()
        except Exception:
            return 0
        try:
            # urgent=False：预取只填空槽位，绝不和当前层抢
            return int(mgr.need(layer_id, list(experts), urgent=False) or 0)
        except Exception:
            return 0

    # -- forward -----------------------------------------------------------
    def _head_logits(self, h: torch.Tensor) -> torch.Tensor:
        # The released ParallelHead projects in fp32. Copying the 129k x 5120 bf16 head to
        # fp32 is 2.65GB, and doing that once per token on an 8GB card sent the allocator
        # into WDDM shared memory (measured: 8.4s/token, 1.3GB of PCIe per step). Chunking
        # the vocab axis keeps the arithmetic bit-identical -- a matmul is row-independent --
        # while the temporary stays at ~0.7GB.
        head = self.head
        if head.dtype == torch.float32:
            return torch.nn.functional.linear(h.float(), head)
        if (getattr(self, "tier", None) is not None
                and self.tier.level < OptLevel.EXTREME
                and __import__("os").environ.get("TI_HEAD_KERNEL") != "0"):
            # the fault this kernel used to hit was a host-resident head (a CPU pointer
            # passed as a device one); with the head placed by the plan and checked below
            # the kernel is exact and beats the chunked fp32 GEMM (0.800 vs 0.835s/token)
            try:
                # cuBLAS 的 bf16 GEMV：实测 1.81ms / 719GB/s，比手写 head_gemv 的
                # 18.19ms / 71GB/s 快 10 倍。累加也是 fp32，输出转回 fp32。
                if (head.device.type == "cuda" and head.dtype == torch.bfloat16
                        and head.is_contiguous()
                        and h.reshape(-1, head.shape[1]).shape[0] == 1):
                    _hb = h.reshape(-1, head.shape[1]).to(torch.bfloat16)
                    _o = torch.nn.functional.linear(_hb, head)
                    return _o.float().view(*h.shape[:-1], head.shape[0])
            except Exception as _exc:
                if __import__("os").environ.get("TI_HEAD_DEBUG"):
                    print(f"[head] kernel path unavailable: {_exc!r}", flush=True)
            hf = h.float()
            step = 32768
            out = None
            for start in range(0, head.shape[0], step):
                chunk = torch.nn.functional.linear(hf, head[start:start + step].float())
                if out is None:
                    out = torch.empty(*chunk.shape[:-1], head.shape[0],
                                      dtype=chunk.dtype, device=chunk.device)
                out[..., start:start + step] = chunk
            return out
        return torch.nn.functional.linear(h.to(head.dtype), head)

    def _forward_chunk(self, ids: torch.Tensor, start_pos: int, positions=None):
        # positions: optional [b] per-row position of each row's last token, for a batch of
        # sequences at different depths. None is the single-sequence path, unchanged.
        self._last_hidden = None  # filled after the final layer (MTP stage-0 input)
        self._target_layers = set(self.cfg.dspark_target_layer_ids)
        self._target_hidden: dict[int, torch.Tensor] = {}
        # v5 streamed-layer double buffering state (lazily initialised; at most two
        # signature groups get flat buffers, the rest fall back to the sync copy).
        if not hasattr(self, "_v5_enabled"):
            # 8GB card: even two signature groups of flat buffers push VRAM past the
            # physical limit (measured 9330MB peak -> WDDM paging), so keep it off.
            self._v5_enabled = bool(int(__import__("os").environ.get("TI_V5", "0")))
            self._v5_stream = None
            self._v5_pending: dict[int, object] = {}
        if self.layer_ids is not None:
            layer_seq = self.layer_ids
            if self.allow_missing_layers:
                avail = self.available_layers()
                layer_seq = [i for i in layer_seq if i < avail]
        else:
            n = self.max_layers
            if self.allow_missing_layers:
                n = min(n, self.available_layers())
            layer_seq = list(range(n))
        _tt0 = __import__("time").perf_counter() if __import__("os").environ.get("TI_LAYER_TIME") else None
        if __import__("os").environ.get("TI_MEM_DBG"):
            _mf, _mt = torch.cuda.mem_get_info()
            print(f"[memdbg] T={ids.shape[1]} free={_mf / 2**20:.0f}MB alloc="
                  f"{torch.cuda.memory_allocated() / 2**20:.0f}MB reserved="
                  f"{torch.cuda.memory_reserved() / 2**20:.0f}MB", flush=True)
        if __import__("os").environ.get("TI_GPUDBG") == "1":
            print(f"[trace] embed start capt={torch.cuda.is_current_stream_capturing()}", flush=True)
        h = self.embed[ids.cpu()].to(self.device) if self.embed_on_cpu else self.embed[ids]
        if __import__("os").environ.get("TI_GPUDBG") == "1":
            print("[trace] embed done", flush=True)
        h = h.unsqueeze(2).repeat(1, 1, self.cfg.hc_mult, 1)
        pre_mix = torch.zeros(h.size(0), h.size(1), self.cfg.hc_mult, dtype=torch.float32, device=h.device)
        pre_mix[:, :, 0] = 1.0
        if __import__("os").environ.get("TI_GPUDBG") == "1" and torch.cuda.is_current_stream_capturing():
            print("[capt] pre_mix done", flush=True)
        hash_ids = None
        if (self.hashes is not None
                and not torch.cuda.is_current_stream_capturing()):
            _h0 = __import__("time").perf_counter() if _tt0 is not None else None
            # tolist() 是 D2H，CUDA graph 捕获期间禁止；捕获时跳过哈希（图的 key 是固定
            # 的 ids/start_pos，本来也不需要它）
            hash_ids = self.hashes(ids.reshape(-1).tolist(), start_pos)
            if _h0 is not None:
                print(f"[time] pre-embed {( _h0 - _tt0)*1e3:.1f}ms hash {(__import__('time').perf_counter()-_h0)*1e3:.1f}ms", flush=True)
        # expert prefetch: learned CPU predictor first, temporal routing as fallback
        pred = None
        tok_now = None
        if (self.enable_prefetch and start_pos > 0
                and not torch.cuda.is_current_stream_capturing()):
            if self.predictor is not None and ids.numel() == 1:
                tok_now = int(ids.reshape(-1)[0].item())
                pred = self.predictor.predict(tok_now)
            source = pred or self._last_routing
            if source:
                # only the first layers: the VRAM slot pool holds ~1-2 layers of top-6,
                # so prefetching all 40 layers evicts every slot before it is consumed
                # (measured: 0 slot hits, every read repeated on the compute thread).
                # The rest is prefetched rolling inside the layer loop below.
                # tier.prefetch_k is how many layers ahead to warm: 0 means the tier
                # asked for no lookahead (prefetch off), 2-3 is what the slot pool
                # sustains. It was a config field nobody read, so every tier behaved
                # like k=2.
                _depth = int(getattr(self.tier, "prefetch_k", 2) or 0)
                for layer_i in layer_seq[:max(0, _depth)]:
                    if source.get(layer_i):
                        self._prefetch_layer(layer_i, source[layer_i])
        self._timings = []
        events = []
        if self._v5_enabled and self._v5_stream is None:
            self._v5_stream = torch.cuda.Stream(device=self.device)
        use_flat = self.device != "cpu" and not self._v5_enabled and self._flat_enabled()
        streamed_seq = [i for i in layer_seq if i in self._stream_layers()] if self._v5_enabled else []
        flat_seq = [i for i in layer_seq if i in self._stream_layers()] if use_flat else []
        flat_pos = {j: k for k, j in enumerate(flat_seq)}
        for pos_i, i in enumerate(layer_seq):
            if torch.cuda.is_current_stream_capturing() and i < getattr(self, "_graph_lo", 0):
                continue
            b = self.block(i)
            streamed = i in self._stream_layers()
            if streamed and self.device != "cpu":
                if use_flat:
                    k = flat_pos[i]
                    # ring size is dynamic: an OOM shed may have shrunk it, so wrap by
                    # the live length instead of a hard-coded 3
                    _ring = len(self._flat_state[1]) if isinstance(self._flat_state, tuple) else 1
                    _ring = max(1, _ring)
                    self._flat_wait(i, k % _ring)
                    if __import__("os").environ.get("TI_NO_FLAT_PREFETCH") != "1":
                        for d in range(1, _ring):
                            if k + d < len(flat_seq):
                                nxt = flat_seq[k + d]
                                if nxt not in self._flat_ev:
                                    self._flat_prefetch(nxt, (k + d) % _ring)
                    if k + 2 < len(flat_seq) and self.enable_prefetch:
                        # RAM prefetch two layers ahead: the read has ~2 layers of compute
                        # time to land, and it never touches the VRAM slot pool
                        ahead = flat_seq[k + 2]
                        r = self._last_routing.get(ahead)
                        if r:
                            self._prefetch_layer(ahead, r)
                elif self._v5_enabled and streamed_seq:
                    try:
                        ev = self._v5_pending.pop(i, None)
                        if ev is not None:
                            torch.cuda.current_stream().wait_event(ev)
                        else:
                            if self._ensure_stream_buffers(i)[0] is None:
                                raise RuntimeError("buffer budget exceeded")  # sync fallback
                            self._stream_copy_sync(i)
                        # prefetch the next streamed layer into the other parity buffer
                        if i in streamed_seq:
                            k = streamed_seq.index(i)
                            if k + 1 < len(streamed_seq):
                                nxt = streamed_seq[k + 1]
                                if (nxt % 2) != (i % 2) and nxt not in self._v5_pending:
                                    self._v5_pending[nxt] = self._stream_prefetch(nxt, self._v5_stream)
                    except Exception:
                        self._v5_enabled = bool(int(__import__("os").environ.get("TI_V5", "0")))
                        b.to(self.device)
                else:
                    b.to(self.device)
            if self.profile:
                ev0 = torch.cuda.Event(enable_timing=True)
                ev0.record()
            # getattr: a non-DeepSeek layer (Qwen's QwenLayer) has no engram attribute
            # at all, and the plain attribute access crashed every Qwen run here.
            if getattr(b, "engram", None) is not None and hash_ids is not None:
                _e0 = __import__("time").perf_counter() if __import__("os").environ.get("TI_LAYER_TIME") else None
                h = b.engram.forward(h, hash_ids[b.engram.hash_index])
                if _e0 is not None:
                    print(f"[time] layer {i} engram {(__import__('time').perf_counter()-_e0)*1e3:.1f}ms", flush=True)
            # official (reference/official/model.py:1267): the MTP head reads the *input* of
            # its target layers, hc-collapsed by a plain mean -- not the attn_norm output
            # and not the attention input. Captured before forward() overwrites h.
            _lt0 = __import__("time").perf_counter() if __import__("os").environ.get("TI_LAYER_TIME") else None
            if __import__("os").environ.get("TI_MEM_TRACE"):
                print(f"[mem] layer {i:2d} before alloc={torch.cuda.memory_allocated()/2**20:.0f}MB "
                      f"reserved={torch.cuda.memory_reserved()/2**20:.0f}MB", flush=True)
            h_in = h
            # EXTREME's skip_layer_pattern: drop this layer's residual contribution.
            # It was parsed into the tier config and never consulted, so the pattern
            # looked active while every layer still ran.
            if i in self._skip_layers:
                pre_mix = h
                continue
            try:
                if __import__("os").environ.get("TI_GPUDBG") == "1" and torch.cuda.is_current_stream_capturing():
                    print(f"[capt] layer {i}", flush=True)
                if torch.cuda.is_current_stream_capturing():
                    # **用 _graph_hi 收口，不是 TI_GRAPH_CHUNK。**
                    # 这里以前是 `i >= TI_GRAPH_CHUNK`：分块捕获时外层把
                    # _graph_lo 设成 0/4/8/.../36，但 break 条件仍然拿 i（绝对层号）
                    # 跟 chunk 比，于是第 2 块起 `i=4 >= 4` 立刻 break —— 每块图只
                    # 捕到 _graph_lo 那一层，剩下 3 层没进图。10 块图里 9 块是残的，
                    # replay 时每块都要重新进出一次 _forward_impl，白白多出开销。
                    # 正确条件：跑完 [_graph_lo, _graph_hi) 就停。
                    if i >= getattr(self, "_graph_hi", 1 << 30):
                        break
                # **每层进场前把 _active_layer 设成它自己。**
                #
                # attention / compressor / indexer 的 state 槽都以 _active_layer 为键
                # （_kv_buf、_graph_kv_buf、wo_a_bf 这些常驻 buffer 都挂在里面）。
                # 那个字段原本只在 __init__ 里设一次，于是整套 state 只有「构造时那个
                # 层号」一个槽：L0 建好 KV buffer，L1..L39 全都读同一块，历史互相覆盖。
                #
                # 实测：图里 L0 的 slot5 是 [0,0,0,0]（压根没写进去），整图解码输出是
                # '<｜begin▁of▁sentence｜># 关于" | 202' 这种噪声，eager 同一 prompt 是
                # 完整中文。eager 之所以「看起来正常」，是因为它每层都重新分配 buf，
                # 而捕获期 buf 是常驻的 —— 同一个坑只有图会踩。
                # **只在捕获期设置。**
                #
                # 非捕获期这条赋值会改掉 streaming 层原本的共享槽语义：多个层
                # 共用一个 attn 对象时，_active_layer 是「谁在跑」，而不是
                # 「构造函数里的层号」。eager 一旦在这里被强制改写成 layer_id，
                # 共享槽就退化成每层独占，_forward_impl 第二次调用直接出 NaN
                # （实测 eager absmax=nan，而同一个 eager 路径不设这行时是完整中文）。
                #
                # 捕获期则必须设：图里每层的常驻 buffer（_graph_kv_buf /
                # _graph_freqs）都挂在 _st() 里，槽位弄混就是 40 层共用一块 KV。
                _al = getattr(b, "layer_id", None)
                if _al is not None and torch.cuda.is_current_stream_capturing():
                    for _mod in (getattr(b, "attn", None), getattr(b, "ffn", None)):
                        if _mod is not None and hasattr(_mod, "_active_layer"):
                            _mod._active_layer = int(_al)
                        _cmp = getattr(_mod, "compressor", None)
                        if _cmp is not None and hasattr(_cmp, "_active_layer"):
                            _cmp._active_layer = int(_al)
                        _ix = getattr(_mod, "indexer", None)
                        if _ix is not None and hasattr(_ix, "_active_layer"):
                            _ix._active_layer = int(_al)
                h, pre_mix = b.forward(h, start_pos, pre_mix, positions=positions)
                _dh = getattr(self, "_dump_h", None)
                if _dh is not None:
                    _dh.append((i, h_in.clone(), h.clone()))
            except Exception:
                try:
                    if torch.cuda.is_current_stream_capturing():
                        raise
                    _fr, _ft = torch.cuda.mem_get_info()
                    _m = (f"vram free={_fr/2**20:.0f}MB/{_ft/2**20:.0f}MB "
                          f"alloc={torch.cuda.memory_allocated()/2**20:.0f}MB")
                except Exception:
                    _m = "mem_get_info failed"
                print(f"[layer-fail] layer {i} streamed={streamed} use_flat={use_flat} "
                      f"device={getattr(b, '_device', '?')} {_m} "
                      f"hc_fn={getattr(b, 'hc_attn_fn', None) is not None and b.hc_attn_fn.device}",
                      flush=True)
                raise
            if use_flat and streamed:
                # the ring slot this layer read is free only once these kernels retire
                _ev_done = torch.cuda.Event()
                _ev_done.record()
                self._flat_done[i] = _ev_done
            # 滚动预取第 k+1 层的专家（搬进槽位，不是只预热 RAM）。
            # 放在这里是因为此刻本层 kernel 已经排进流里、还在算，正好的空档：
            # 搬运和计算重叠，下一层的 MoE 就不用等 IO。
            # 预测用上一轮的路由 —— 相邻 token 的专家集合重合度很高，命中就是
            # 白赚；没中也只是多搬一次后面会被覆盖，不额外消耗算力。
            if (self.enable_prefetch and pos_i + 1 < len(layer_seq)
                    and not torch.cuda.is_current_stream_capturing()):
                _nxt = layer_seq[pos_i + 1]
                _pred = self._last_routing.get(_nxt)
                if _pred:
                    try:
                        self._warm_next_layer(_nxt, _pred)
                    except Exception:
                        pass
            if __import__("os").environ.get("TI_DEBUG_NAN"):
                _fin = bool(torch.isfinite(h).all())
                if (not _fin) or __import__("os").environ.get("TI_DEBUG_NAN") == "all":
                    print(f"[dbg] layer {i:2d} finite={_fin} absmax={float(h.abs().max()):.2f}",
                          flush=True)
            if getattr(self, "_layer_dump", None) is not None:
                # diagnostics: per-position activation magnitude after each layer, so a
                # chunked forward and the same tokens one at a time can be diffed layer
                # by layer (probe_layer_dump.py)
                self._layer_dump.append((i, h.float().amax(dim=(0, 2, 3)).tolist()))
            if i in self._target_layers:
                self._target_hidden[i] = h_in.mean(dim=2).detach()
            if self.profile:
                ev1 = torch.cuda.Event(enable_timing=True)
                ev1.record()
                events.append((i, ev0, ev1))
            if _lt0 is not None:
                _lt1 = __import__("time").perf_counter()
                print(f"[time] layer {i} total {(_lt1-_lt0)*1e3:.1f}ms "
                      f"streamed={streamed} use_flat={use_flat}", flush=True)
            if b.ffn._last_ids:
                self._last_routing[i] = b.ffn._last_ids
            if streamed and self.device != "cpu" and not use_flat:
                b.to("cpu")  # free VRAM for the next streamed layer
        if self.predictor is not None and tok_now is not None and self._last_routing:
            self.predictor.observe(tok_now, dict(self._last_routing))
        self._last_hidden = h  # [1, T, hc, dim] before collapse (MTP main hidden)
        # official DSparkBlock.forward: at start_pos == 0 a draft stage only *seeds* its
        # main-KV window (attn(x, 0, main_x)); without this the draft attends over an
        # empty window and its predictions are noise (measured accept 3% instead of ~50%)
        if start_pos == 0 and getattr(self, "mtp", None) is not None:
            try:
                _th = sorted(getattr(self, "_target_hidden", {}) or {})
                _want = list(self.cfg.dspark_target_layer_ids)
                if len(_th) >= len(_want):
                    _mx = self.mtp[0].main_hiddens([self._target_hidden[k] for k in _th])
                    for _st in self.mtp:
                        _seed = _mx.new_zeros(1, self.cfg.dspark_block_size, self.cfg.dim)
                        _st.attn.forward_with_main(_seed, 0, _mx)
            except Exception as _exc:
                print(f"[tinyinfer] MTP seed skipped: {_exc!r}", flush=True)
        self._last_pre_mix = pre_mix  # exact HC mixing used for the collapse
        if self.profile:
            torch.cuda.synchronize()
            self._timings = [(i, a.elapsed_time(b)) for i, a, b in events]
        h = (pre_mix.unsqueeze(-1) * h.float()).sum(dim=2).to(h.dtype)
        h = rms_norm(h, self.out_norm, self.eps)
        if self.head_on_cpu:
            h = h.cpu()
        return self._head_logits(h)

    def prefill_layer_major(self, ids: torch.Tensor, max_chunk: int | None = None) -> torch.Tensor:
        """Layer-major prefill: stream each layer once for the whole prompt.

        The chunk-major path (forward) re-streams all 40 layers per chunk, so a long
        prompt pays the layer H2D and the expert reads once per chunk (measured
        prefill 64 = 314s at chunk 8, i.e. hours for 128K). Here the loop order is
        swapped: activations for every chunk sit in host RAM, each layer is moved to
        the device once, applied to all chunks, then freed. Only the activation list
        grows with the prompt (T x hc x dim x 2B = 5.2GB at 128K, fits in 47GB).
        """
        if ids.device.type != self.device and self.device != "cpu":
            ids = ids.to(self.device)
        b, T = ids.shape
        step = max_chunk or self.prefill_chunk
        # per-kv-source handoff slots for the whole pass (blocks are constructed lazily,
        # so the flag has to be set before any block() call below).
        self._layer_major_mode = True
        self._rt_cache = {}
        try:
            return self._prefill_layer_major_impl(ids, T, step)
        finally:
            self._layer_major_mode = False
            self._rt_cache = {}

    def _prefill_layer_major_impl(self, ids, T: int, step: int):
        if not hasattr(self, "_v5_enabled"):
            self._v5_enabled = bool(int(__import__("os").environ.get("TI_V5", "0")))
        # layer sequence identical to _forward_chunk
        if self.layer_ids is not None:
            layer_seq = self.layer_ids
            if self.allow_missing_layers:
                avail = self.available_layers()
                layer_seq = [i for i in layer_seq if i < avail]
        else:
            n = self.max_layers
            if self.allow_missing_layers:
                n = min(n, self.available_layers())
            layer_seq = list(range(n))
        # 1) embed every chunk once; keep the activation on the host between layers
        acts = []
        for off in range(0, T, step):
            chunk_ids = ids[:, off:off + step]
            h = self.embed[chunk_ids.cpu()].to(self.device) if self.embed_on_cpu else self.embed[chunk_ids]
            h = h.unsqueeze(2).repeat(1, 1, self.cfg.hc_mult, 1)
            pre_mix = torch.zeros(h.size(0), h.size(1), self.cfg.hc_mult, dtype=torch.float32, device=h.device)
            pre_mix[:, :, 0] = 1.0
        if __import__("os").environ.get("TI_GPUDBG") == "1" and torch.cuda.is_current_stream_capturing():
            print("[capt] pre_mix done", flush=True)
            hash_ids = None
            if self.hashes is not None:
                hash_ids = self.hashes(chunk_ids.reshape(-1).tolist(), off)
            acts.append([h.cpu(), pre_mix.cpu(), hash_ids, off])
        # 2) per layer: one H2D, apply to every chunk, free. Cross-layer shared state in
        # compressed attention (shared_attn.candidates/compress_kv/index_k) is published
        # by the kv_source layer and consumed by its followers, so it must be cleared at
        # every layer boundary in layer-major order (chunk-major keeps them within one
        # layer's chunk sequence, which is why it never needed this).
        self._timings = []
        for i in layer_seq:
            bobj = self.block(i)
            streamed = i in self._stream_layers()
            if streamed and self.device != "cpu":
                bobj.to(self.device)
            elif self.device != "cpu":
                bobj.to(self.device)
            for slot in acts:
                h_cpu, pre_cpu, hash_ids, off = slot
                h = h_cpu.to(self.device, non_blocking=False)
                pre_mix = pre_cpu.to(self.device, non_blocking=False)
                if getattr(bobj, "engram", None) is not None and hash_ids is not None:
                    h = bobj.engram.forward(h, hash_ids[bobj.engram.hash_index])
                h, pre_mix = bobj.forward(h, off, pre_mix)
                slot[0] = h.cpu()
                slot[1] = pre_mix.cpu()
            if getattr(bobj.ffn, "_last_ids", None):
                self._last_routing[i] = bobj.ffn._last_ids
            if streamed and self.device != "cpu":
                bobj.to("cpu")
        # 3) final head on the last chunk
        h_cpu, pre_cpu, _, _ = acts[-1]
        h = h_cpu.to(self.device, non_blocking=False)
        pre_mix = pre_cpu.to(self.device, non_blocking=False)
        self._last_hidden = h
        h = (pre_mix.unsqueeze(-1) * h.float()).sum(dim=2).to(h.dtype)
        h = rms_norm(h, self.out_norm, self.eps)
        if self.head_on_cpu:
            h = h.cpu()
        return self._head_logits(h)

    # -- VRAM pressure + CUDA graph + prefix checkpoints ---------------------
    def _shed_vram(self) -> None:
        """Give memory back after an OOM: shrink the streaming ring, drop the slot pool."""
        st = getattr(self, "_flat_state", None)
        if st is not None and st != "off":
            ring = list(st[1])
            if len(ring) > 2:
                # keep the two buffers decode needs and actually release the rest: a
                # shorter list alone does not give the VRAM back
                keep = ring[:2]
                dropped = ring[2:]
                self._flat_state = (st[0], keep, st[2])
                for buf in dropped:
                    del buf
                print(f"[tinyinfer] CUDA OOM: streaming ring 3 -> {len(keep)} buffers "
                      f"(released {len(dropped)})", flush=True)
        try:
            if getattr(self, "expert_store", None) is not None:
                self.expert_store.drop_slots()
        except Exception:
            pass
        if self.device != "cpu":
            torch.cuda.empty_cache()

    def _graph_capture_shape(self, ids: torch.Tensor, start_pos: int) -> bool:
        # 不用 CUDA graph。tier 里虽然还留着一个 cuda_graph 标志，但引擎这边
        # graph 由 tier.cuda_graph 控制；引擎这边不再无条件关死。
        # unavailable"），而它每次尝试都会 snapshot 整个 KV、跑一次完整 forward、
        # 再 synchronize。关掉之后 torch 也不会在每次 op 里去查
        # cudaStreamIsCapturing（实测每层 8 次）。
        # 暂时关死：捕获期我们的 scratch / y_buf / KV 环形槽都不在 graph 的
        # 内存池里，捕获时对它们做的 DtoD 落到非法地址，会打出粘性
        # CUDA_ERROR_ILLEGAL_ADDRESS 把整个 context 弄坏（不是捕获失败那么轻）。
        # 要开必须先把这些 buffer 全部改成「捕获前预分配、捕获期只写不分配」。
        # 捕获期禁止 H2D 互拷，而慢路径里就有 indices.tolist()。纯 GPU 路径
        # 不含这种调用，所以只在槽位表就绪（pin 之后）才允许捕获。
        # 捕获期会打出粘性 CUDA_ERROR_ILLEGAL_ADDRESS（cuMemcpyDtoDAsync_v2），
        # 把整个 context 弄坏，比不 graph 更糟。当前 attention/hc 段有分配和
        # 环形槽写入不在 graph pool 里，必须先把它们改成「捕获前预分配、
        # 捕获期只写不分配」才能开。在那之前关死。
        _slots0 = getattr(getattr(self, "expert_store", None), "slots", None)
        if getattr(_slots0, "_slot_tab", None) is None:
            return False
        if not getattr(self.tier, "cuda_graph", False):
            return False
        if self.device == "cpu" or ids.shape[1] != 1:
            return False
        if self.gpu_layers < self.max_layers:
            return False
        # 捕获期禁止任何 H2D 互拷，而慢路径的 serve() 里就有一次
        # indices.tolist()（D2H）。纯 GPU 路径（_forward_gpu_only）不含这种
        # 调用，所以只在槽位表已经建好、也就是 pin 之后才允许捕获 ——
        # warmup 阶段捕到的图也会因为槽位表内容为空而不可用。
        _slots = getattr(getattr(self, "expert_store", None), "slots", None)
        if getattr(_slots, "_slot_tab", None) is None:
            return False
        if getattr(self, "mtp", None) is not None:
            return False
        return True

    def _graph_run(self, ids: torch.Tensor, start_pos: int):
        g = getattr(self, "_graph", None)
        if g is None:
            return None
        out = g["out"]
        if out.shape[0] != ids.shape[0]:
            return None
        # **图是「一次一个 token」的图，形状必须逐维对齐。**
        #
        # 只比 batch 维会漏掉 seqlen：prefill 送进来 (1,7) 时图里那块输入是 (1,1)，
        # copy_ 直接抛
        #   RuntimeError: output with shape [1, 1] doesn't match the broadcast shape [1, 7]
        # 而调用方把这次异常一路上抛成了「槽位表就绪但图不存在」。
        #
        # 实测：warmup 之后第一次 decode 建好图（pos=5），紧接着换一个 7 token 的
        # prompt 就炸；同一进程里连续问第二句话必挂。
        if g["in"].shape != ids.shape:
            return None
        # 位置相关的 cos/sin 在每步 replay 前重写（图内不会重跑 Python 写入）
        try:
            from .deepseek.compressed_attention import refresh_cos_sin
            _blocks = list(getattr(self, "_blocks", {}).values())
            # **每一层都要刷，不是只刷第 0 层。**
            # freqs_cis 由 get_table 按 _rope_key 缓存，不同层可能拿到不同的表；
            # 而图里引用的是各层自己那张表的 device 地址。只刷 _blocks[0] 时，
            # 其余层在 replay 里用的是捕获那一刻的 cos/sin —— 位置对不上，
            # 注意力权重全错。实测症状：L0 单独捕获数值逐位正确，加进 L1 之后
            # 整张图的输出恒为 0.0（精确的 0，不是垃圾值）。
            # **按 attn 实例刷，不能按表去重。**
            #
            # 40 层共用同一张 freqs_cis（get_table 按 _rope_key 缓存），所以
            # 按 data_ptr 去重只会刷到第一层。但每层的 _graph_freqs 是各自独立的
            # 常驻 buffer（图里引用的是各层自己那块地址），只刷一层等于其余 39 层
            # 的 RoPE 一直停在捕获那一刻的位置上 —— 位置越往后错得越多。
            _seen_tbl = set()
            for _b in _blocks:
                _att = getattr(_b, "attn", None)
                _tbl = getattr(_att, "freqs_cis", None)
                if _tbl is None:
                    continue
                _k = int(_tbl.data_ptr())
                if _k not in _seen_tbl:
                    _seen_tbl.add(_k)
                    refresh_cos_sin(_tbl, start_pos, int(ids.shape[1]))
                # 窗口层（MLAWindowAttention）捕获期把 freqs 换成了挂在 self 上的
                # 常驻 _graph_freqs（普通切片会被图钉死成捕获那一刻的地址）。
                # 这里把当前 position 的频率覆盖进去 —— 每层各自刷。
                _lid = getattr(_b, "layer_id", getattr(_att, "layer_id", None))
                _stl = getattr(_att, "_states", {}).get(_lid) if _lid is not None else None
                _gf = None if not _stl else _stl.get("_graph_freqs")
                if _gf is not None:
                    _n = int(_gf.shape[0])
                    if _n > 0 and start_pos + _n <= _tbl.shape[0]:
                        _gf.copy_(_tbl[start_pos:start_pos + _n])
                # 窗口索引用的位置 buffer：图外写值，图内读同一块地址。
                from . import nn as _nnmod
                if getattr(_nnmod, "_POS_BUF", None) is not None:
                    _nnmod._POS_BUF.fill_(int(start_pos))
        except Exception:
            pass
        g["in"].copy_(ids)
        for _gr in g["graphs"]:
            _gr.replay()
        return out.clone()

    def _pregrow_graph_buffers(self, ids: torch.Tensor) -> None:
        """把捕获期要用的常驻 buffer 提前建好，避开 graph pool。

        hc.py 里那几个环（hc_pre_rms_norm 的 out / pre_mix、hc_post 的 out、
        hc_split_sinkhorn 的 pre/post/comb）原本是在 is_current_stream_capturing()
        为 True 时第一次分配的。那等于把它们分配进了 graph pool —— pool 认为这些
        是「可以复用的中间张量」，几次 kernel 之后同一块地址就被别的算子拿走，
        于是同一个地址先被写成 A、再被读成 B。

        实测：hc_pre_rms_norm 写进 12922255360 的是 [0.032715, -0.005859, ...]，
        紧接着 wqkv 从同一个地址读到 [0.039551, 0.011047, ...]，差 0.017；一路传到
        最终 argmax 都不同。eager 没这问题，因为它每次都是新显存。

        在捕获窗口外先按捕获期的形状各建一次，地址就落在普通显存上。
        """
        from .deepseek import hc as _hc

        # **每层的 KV 环形 buffer 也必须在窗口外建好。**
        #
        # 图里的 attention kernel 会把这块 buffer 的地址写死进参数。它如果是在
        # 捕获窗口内第一次分配的，就落进 graph pool；而 prefill 走
        # _forward_impl（窗口外），拿到的可能是另一块地址 —— 于是「图 replay 读
        # 的 KV」和「prefill 写的 KV」是两套 cache，历史一个字节都不通。
        #
        # 这里提前按 [1, window, head_dim] 建一次并挂到该层的 state 上，捕获期
        # 就会直接复用这块普通显存。
        try:
            _win = int(self.cfg.sliding_window)
            _hd = int(self.cfg.head_dim)
            for _lid, _blk in getattr(self, "_blocks", {}).items():
                _att = getattr(_blk, "attn", None)
                if _att is None:
                    continue
                _old = getattr(_att, "_active_layer", None)
                try:
                    _att._active_layer = int(_lid)
                    _cur = _att._kv_buf
                    if (_cur is None or _cur.shape[0] != 1 or _cur.shape[1] != _win
                            or _cur.shape[2] != _hd):
                        _att._kv_buf = torch.zeros(1, _win, _hd, dtype=torch.bfloat16,
                                                   device=self.device)
                finally:
                    if _old is not None:
                        _att._active_layer = _old
        except Exception:
            pass

        # **压缩 KV 池也必须预建到足够大。**
        #
        # 捕获期 compress_kv 的宽度会进图的 kernel 参数（切片/掩码都按它算）。
        # 池子在 decode 期间一旦扩容，_full 就换了一块新显存，图里记的还是旧地址 ——
        # replay 读到的是已被回收的内容。这里按 max_position_embeddings // ratio
        # 一次性建满，decode 期间不再增长。
        try:
            _mx = int(getattr(self.cfg, "max_position_embeddings", 0) or 0)
            _ratios = list(getattr(self.cfg, "compress_ratios", ()) or ())
            _hdim = int(self.cfg.head_dim)
            for _lid, _blk in getattr(self, "_blocks", {}).items():
                _att = getattr(_blk, "attn", None)
                if _att is None or not _ratios or _lid >= len(_ratios):
                    continue
                _rt = int(_ratios[_lid])
                if _rt <= 0 or _mx <= 0:
                    continue
                # 上限 4096 组：1M 上下文按 ratio 2 算是 500k 组、每层 512MB，
                # 40 层直接吃满整卡。4096 组足够覆盖 8k token 以内的 decode，
                # 更长的上下文回落到「池子扩容 → 重新捕获」（见 _graph_run 里的形状检查）。
                _need = max(1024, min(4096, _mx // _rt))
                _old = getattr(_att, "_active_layer", None)
                try:
                    _att._active_layer = int(_lid)
                    _cur = _att._compress_kv
                    if _cur is None or _cur.shape[1] < _need:
                        _att._compress_kv = torch.zeros(1, _need, _hdim,
                                                        dtype=torch.bfloat16,
                                                        device=self.device)
                finally:
                    if _old is not None:
                        _att._active_layer = _old
        except Exception:
            pass

        b, s = int(ids.shape[0]), int(ids.shape[1])
        d = int(self.cfg.dim)
        hc_mult = int(self.cfg.hc_mult)
        dev = self.device
        dtype = torch.bfloat16

        # 每个环预先备好和捕获期同样数量的块：捕获期按调用顺序轮转取值，
        # 这里只要保证第一次分配发生在窗口外，后续复用就是同一批地址。
        for _ in range(2):
            _hc._CAP_PRE_OUT.setdefault(
                (b * s, d, dtype),
                [torch.empty((b * s, d), dtype=dtype, device=dev) for _ in range(64)])
            _hc._CAP_PRE_MIX.setdefault(
                (b * s, hc_mult),
                [torch.empty((b * s, hc_mult), dtype=torch.float32, device=dev)
                 for _ in range(64)])
            _hc._CAP_POST_OUT.setdefault(
                (b * s, hc_mult, d, dtype),
                [torch.empty((b * s, hc_mult, d), dtype=dtype, device=dev)
                 for _ in range(64)])
            _hc._CAP_MIX_OUT.setdefault(
                (b, s, hc_mult, dtype),
                [(torch.empty((b, s, hc_mult), dtype=dtype, device=dev),
                  torch.empty((b, s, hc_mult), dtype=dtype, device=dev),
                  torch.empty((b, s, hc_mult, hc_mult), dtype=dtype, device=dev),
                  torch.empty((b, s, hc_mult, hc_mult), dtype=dtype, device=dev))
                 for _ in range(64)])
        # 游标也要一起备好：环是"预建"的，但索引表若不存在，捕获期第一次取用
        # 会在 _CAP_*_IDX[key] 上抛 KeyError。
        for _k, _v in _hc._CAP_PRE_OUT.items():
            _hc._CAP_PRE_IDX.setdefault(_k, 0)
        for _k, _v in _hc._CAP_PRE_MIX.items():
            _hc._CAP_PRE_MIX_IDX.setdefault(_k, 0)
        for _k, _v in _hc._CAP_POST_OUT.items():
            _hc._CAP_POST_IDX.setdefault(_k, 0)
        for _k, _v in _hc._CAP_MIX_OUT.items():
            _hc._CAP_MIX_IDX.setdefault(_k, 0)
        torch.cuda.synchronize()

    def _graph_force_load_routes(self, slots) -> None:
        """把「已经路由过、但此刻不在槽位」的专家全部装进来。

        CUDA graph 里的 MoE 只认 tab[layer][expert] 查到的槽位号直通 kernel，捕获期
        又跳过 _miss_scan / need / commit（都要 D2H，图内做不了）。所以只要表里有
        一个 -1，kernel 就会拿 -1 * blob_bytes 当偏移读到别的专家的字节 —— 实测
        图内 L1 的 sids = [7,6,11,-1,-1,13]、输出 absmax 2.36e37，而 eager 正常。

        这里按 MoE 自己累计的 _route_counts（哪些专家被路由过）逐个补齐，比按
        「覆盖率挑热门」更贴近这一轮真正会用到的集合。
        """
        if slots is None:
            return
        total = 0
        for lid, blk in getattr(self, "_blocks", {}).items():
            rc = getattr(getattr(blk, "ffn", None), "_route_counts", None)
            if not rc:
                continue
            # **按这一轮实际用到的 topk 取，不是「历史上路由过的全部」。**
            #
            # _route_counts 是 warmup + 前面若干 token 累积出来的，一层可能是几十个
            # 专家。全装进来会瞬间吃满 1057 个槽位，装不下的就留 -1 —— 而图里的 MoE
            # 没有 miss 扫描，那些专家直接被跳过，和 eager 的参与集合不一致。
            #
            # 实测：L0 的输入相同、hc 两次归一化也相同，但 L0 的输出 x 不同
            # （eager 0.2334 vs graph 0.23242，第二维 -0.01782 vs 0.06982），
            # 一路传到 L1 的 attn、gate 路由、最终 argmax。
            #
            # 图内一次 forward 每层最多路由 topk 个专家，按计数从高到低取这么多就够，
            # 而且取的是「这一轮最可能用到的」——正是 eager 也会加载的那批。
            topk = int(getattr(blk.ffn, "topk", 0) or 0)
            ranked = sorted(rc.items(), key=lambda kv: -int(kv[1]))
            want = [int(e) for e, _c in ranked[:max(topk, 1)]]
            need = [e for e in want if not slots.has(lid, e)]
            if not need:
                continue
            try:
                self.expert_store.fetch(lid, need)
            except Exception:
                pass
            for e in need:
                if slots.has(lid, e):
                    continue
                try:
                    slots.acquire(lid, e)
                    total += 1
                except Exception:
                    # 槽位真的满了：这里不能静默 —— 图里那些专家一定是 -1。
                    break
        if total:
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            try:
                slots.table_sync()
                torch.cuda.synchronize()
            except Exception:
                pass

    def _restore_active_layers(self) -> None:
        """把每个 attention 模块的 _active_layer 还原成它构造时的层号。

        捕获期 _forward_chunk 会把它改成「正在跑的那一层」，因为图里的常驻
        buffer（_graph_kv_buf / _graph_freqs）都挂在 _st() 里、按 _active_layer
        取槽。捕获结束必须还原：restore() 会逐层写 a._kv_window / a._kv_buf，
        那两个 property 也走 _st()，键还是 _active_layer；停在上次的层号上，
        40 层的 KV 历史会全写进同一层的槽，其余层继续引用图专用 buffer ——
        实测捕获后紧接着跑 eager，absmax=nan、argmax=0，而同一个 eager 在没
        捕获过的进程里是完整通顺的中文。
        """
        for _bb in getattr(self, "_blocks", {}).values():
            for _mm in (getattr(_bb, "attn", None), getattr(_bb, "ffn", None)):
                if _mm is None:
                    continue
                for _sub in (_mm, getattr(_mm, "compressor", None),
                             getattr(_mm, "indexer", None)):
                    if _sub is None:
                        continue
                    _lid = getattr(_sub, "layer_id", None)
                    if _lid is not None:
                        _sub._active_layer = int(_lid)

    def _graph_try_capture(self, ids: torch.Tensor, start_pos: int) -> None:
        if getattr(self, "_graph", None) is not None or getattr(self, "_graph_failed", False):
            return
        if not self._graph_capture_shape(ids, start_pos):
            return
        try:
            # 捕获期不能新建 tensor：window_idxs_ring 用的全局位置 buffer 先建好。
            try:
                from . import nn as _nn
                import torch as _t
                if _nn._POS_BUF is None or _nn._POS_BUF.device != _t.device("cuda"):
                    _nn._POS_BUF = _t.zeros((), dtype=_t.long, device="cuda")
            except Exception:
                pass
            # warmup 也要用同一个位置：窗口索引从 _POS_BUF 读，旧值会让图内图外不一致。
            try:
                from . import nn as _nn2
                if getattr(_nn2, "_POS_BUF", None) is not None and not torch.cuda.is_current_stream_capturing():
                    _nn2._POS_BUF.fill_(int(start_pos))
            except Exception:
                pass
            _pinned_on = []
            # **捕获期要用的"常驻 buffer"必须在进入捕获窗口之前分配。**
            #
            # hc.py 里那些环是在 is_current_stream_capturing() 为 True 时第一次建的
            # —— 也就是说它们本身就落在 graph pool 里，pool 把它们当成可复用的中间
            # 张量，几个 kernel 之后同一块地址就归了别人。
            # 实测：hc_pre_rms_norm 写进 12922255360 的是 [0.032715, ...]，紧接着
            # wqkv 从同一个地址读到 [0.039551, ...]。
            #
            # 这里在窗口外预热一次，让地址落到普通显存上。
            try:
                self._pregrow_graph_buffers(ids)
            except Exception:
                pass
            snap = self.snapshot()
            # 捕获流程内部的 restore(snap) 必须保留图内 buffer（见 restore 里的注释）
            self._graph_just_captured = True
            _keep_rt = None   # finally 里要用，必须在 try 之外初始化
            try:
                in_buf = ids.clone()
                self._forward_impl(in_buf, start_pos, None)
                torch.cuda.synchronize()
                # **warmup 之前必须把所有层的权重都搬上 device。**
                #
                # 量化路径下 wq_a / wkv 这类小矩阵是懒搬的（QuantLinear 在
                # forward 里做 .to(device)）。那意味着捕获期第一次搬运会新建一块
                # 显存，其地址被写进图中的 kernel 参数；捕获结束后那块临时内存
                # 被 allocator 回收，replay 时读到的就是别人的字节 —— 实测 L1
                # 输出 absmax 2.36e37，正是 bf16 把任意 16 位当数值解释的量级。
                #
                # eager 每步都重新搬一次，所以永远看不出问题；layers=[0,0] 也是
                # 对的（同一个 Block，第二次已经是热权重）。只有「不同层 + graph」
                # 这个组合才暴露。
                _moved = []
                for _b in self._blocks.values():
                    try:
                        _r = _b.to(self.device)
                        if _r is not None:
                            _moved.append(_b)
                    except Exception:
                        pass
                if _moved:
                    torch.cuda.synchronize()
                # **warmup 这次带 miss 扫描跑，把本步真实需要的专家装进槽位、写进表。**
                #
                # 图里的 MoE 只认 tab[layer][expert] 查到的槽位号直通 kernel，而
                # 捕获期会跳过 _miss_scan / need / commit（要 D2H，图内做不了），
                # 表里的 -1 就永远补不上，kernel 拿到 -1 是非法地址 —— 实测 L20
                # 上 3 层正确、第 5 层起 NaN，就是这个原因。
                #
                # 所以这里只有一件事：确保刚跑完的这一次把表落定。不再打开
                # _all_pinned —— 它会让 MoE 跳过扫描，而扫描正是填表的唯一途径。
                _sl_cap = getattr(getattr(self, "expert_store", None), "slots", None)
                if _sl_cap is not None:
                    try:
                        torch.cuda.synchronize()
                        _sl_cap.slot_table(int(self.max_layers), 384)
                        if hasattr(_sl_cap, "table_sync"):
                            _sl_cap.table_sync()
                        torch.cuda.synchronize()
                    except Exception:
                        pass
                    # **捕获这一轮要路由到的专家，必须此刻全部在槽位里。**
                    #
                    # 图里的 MoE 只认 tab[layer][expert] 查到的槽位号直通 kernel，而
                    # 捕获期会跳过 _miss_scan / need / commit（都要 D2H，图内做不了）。
                    # 所以只要有一个专家此刻不在槽位，表里就是 -1，kernel 拿
                    # -1 * blob_bytes 当偏移，读到的是别的专家的字节 —— 实测图内 L1
                    # 的 sids = [7,6,11,-1,-1,13]，输出 absmax 2.36e37，而 eager 走
                    # 同一套权重结果正常。
                    #
                    # pin_hot_experts 是按覆盖率挑的热门专家，不保证覆盖这一轮的
                    # 真实路由，所以这里按「刚跑完这次 forward 记录到的路由」补装。
                    self._graph_force_load_routes(_sl_cap)
                # **捕获必须从和 eager 相同的干净状态开始。**
                #
                # 上面那次 forward 是 warmup：它推进了 KV、填了 _decode_idxs、更新了
                # 各层的 _kv_len。捕获紧接着跑，看到的是「已经走过一步」的残局 ——
                # 而 eager 是从 reset() 后的零状态跑的。两边起点不同，L0 的 attention
                # 就取出不同的值（实测 blk_out 1.055 vs 1.35），一路传到 MoE 的 gate，
                # 路由从 [79,132,292,278,91,372] 漂成 [132,79,309,349,348,372]，
                # 后两个从未进过槽位，表里是 -1。
                #
                # 所以 warmup 之后把 KV 状态清回「一次都没跑过」——那正是 eager
                # 在 reset() 之后的起点。直接用 restore(snap) 不够：snap 是 warmup
                # 之前拍的，里面 kv 本来就是 None，但 restore 对 None 只是不赋值，
                # 捕获期新建的 buffer 会留在 _states 里（_keep_buf 分支之外的分支
                # 才 clone）。这里显式清干净，任何残留都不带进捕获。
                # 注意**不能**碰 _rt：warmup 建好的跨层状态稍后要放回去。
                self.restore(snap)
                for _bq in self._blocks.values():
                    _aq = getattr(_bq, "attn", None)
                    if _aq is not None and hasattr(_aq, "reset"):
                        try:
                            _aq.reset()
                        except Exception:
                            pass
                # **捕获前这次完整 forward 产出的 _rt 必须留住。**
                #
                # 分块捕获时每块只跑 [_graph_lo, _graph_hi) 那几层，前面层的
                # continue 掉。但 compress_kv / index_k / topk_idxs / candidates
                # 是 kv_source 层（L2/L8/L14/L20）产出的**跨层共享状态**，后面
                # 34 层都直接读它。第 2 块起既然跳过了 L2，index_k 就没人建，
                # 于是随后的层拿到 None：
                #   compressor.py:285  index_k = self.rt.index_k[:, :t]
                #   TypeError: NoneType object is not subscriptable
                # 以前这里被 except 吞掉、静默回落 eager，所以看不出来；删掉
                # 回落之后它立刻现形。
                #
                # 上面这次完整 forward 已经把 _rt 建好了，别让后面的 restore(snap)
                # 把它清回 None（snap 是跑之前拍的，里面 _rt 就是 None）。
                try:
                    from .deepseek.compressor import shared_attn as _sa_k
                    _keep_rt = {
                        "compress_kv": getattr(_sa_k, "compress_kv", None),
                        "index_k": getattr(_sa_k, "index_k", None),
                        "topk_idxs": getattr(_sa_k, "topk_idxs", None),
                        "candidates": getattr(_sa_k, "candidates", None),
                        "candidates_by_chunk": dict(getattr(_sa_k, "candidates_by_chunk", {}) or {}),
                        "src_ratio": getattr(_sa_k, "src_ratio", None),
                    }
                except Exception:
                    _keep_rt = None
                # 分块捕获：每 TI_GRAPH_CHUNK 层一张图，所有图共享同一个 graph pool。
                # 块与块之间的 h/pre_mix 都在 pool 内、地址固定，replay 时依次执行。
                # **默认整段捕获，不再每 4 层切一块。**
                #
                # 分块捕获要求每块图能从上一块的中间状态接着算，但 _forward_impl
                # 每次都从 embed 开始：第 2 块跳过 L0-3 之后，h 拿到的仍是
                # embedding 输出而不是 L3 的输出，于是整条链从第 5 层起就
                # 算的是错的东西 —— 实测逐块 replay 时第 0 块就已经是 NaN。
                #
                # 要真正分块，得让块与块之间显式传递 h/pre_mix（各存一块固定
                # buffer，块内只写它们）。在那之前，一块图跑完全部 40 层是唯一
                # 正确的做法：捕获一次，replay 一次。
                _env_chunk = int(__import__("os").environ.get("TI_GRAPH_CHUNK", "0") or 0)
                if _env_chunk <= 0 or _env_chunk >= self.max_layers:
                    _chunk = int(self.max_layers)
                else:
                    _chunk = _env_chunk
                _pool = torch.cuda.graph_pool_handle()
                _graphs = []
                out = None
                for _lo in range(0, self.max_layers, _chunk):
                    self._graph_lo = _lo
                    self._graph_hi = min(_lo + _chunk, self.max_layers)
                    _g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(_g, pool=_pool):
                        out = self._forward_impl(in_buf, start_pos, None)
                    _graphs.append(_g)
                torch.cuda.synchronize()
                # 图已经把捕获期那些 buffer 的地址记进 kernel 参数了；从这一刻起
                # restore 不能再换 buffer（见 restore 里的 KEEP 分支）。
                self._graph_captured_done = True
                # 环的游标只对「捕获那一遍」有意义；下一张图（或下一轮进程内的
                # 重新捕获）必须从 0 开始，否则新图里每个 kernel 绑到的 buffer
                # 和上一次捕获的位置错开，两张图共用池时会互相踩。
                try:
                    from .deepseek import hc as _hcm
                    for _nm in ("_CAP_PRE_IDX", "_CAP_PRE_MIX_IDX", "_CAP_POST_IDX",
                                "_CAP_MIX_IDX"):
                        getattr(_hcm, _nm).clear()
                except Exception:
                    pass
            finally:
                # **必须先还原 _active_layer，再 restore(snap)。**
                #
                # restore 里对每一层写 a._kv_window / a._kv_buf，而那两个 property
                # 都走 _st()，键就是 _active_layer。捕获期最后一层是 L39，若不先
                # 还原，40 层的 KV 历史会全部写进 L39 的 state 槽，其余 39 层的
                # _kv_buf 保持捕获期那块图专用 buffer —— eager 接着用就是 NaN。
                self._restore_active_layers()
                self.restore(snap)
                self._graph_just_captured = False
                self._graph_captured_done = False
                # **捕获期把 _active_layer 逐层改成 layer_id，必须还原。**
                #
                # _st() 以 _active_layer 为键，所有常驻 buffer（_kv_buf /
                # _graph_freqs / wo_a_bf）都挂在里面。捕获结束后如果不还原，
                # 每个模块的 _active_layer 都停在最后一层的编号上，此后 eager
                # 的每一层都去读写同一层的 state —— 实测捕获过图之后紧接着跑
                # eager，absmax=nan、argmax=0。而同一个 eager 在没捕获过的进程里
                # 是完整通顺的中文。
                #
                # 还原成构造时的层号（每个模块自己记着的那一个）。
                # 把捕获前那次完整 forward 建好的跨层共享状态放回去（见上面
                # _keep_rt 的注释）。restore(snap) 会把 _rt 清成 None，那会让
                # 随后 replay 出来的图在 compressor 里拿到 None。
                if _keep_rt is not None:
                    try:
                        from .deepseek.compressor import shared_attn as _sa_r
                        for _k, _v in _keep_rt.items():
                            setattr(_sa_r, _k, _v)
                    except Exception:
                        pass
            # **pool 必须一直被引用。** 捕获期分配的张量（含 out 本身）都住在
            # graph pool 里；pool_handle 只是句柄，真正的存储由 CUDAGraph 持有。
            # 实测：不保留 pool 时 n>=2 的输出恒为 0.0（kernel 都执行了，但写进了
            # 已被回收的地址），n=1 却能对 —— 因为单层图不会触发 pool 的重新分配。
            self._graph = {"graphs": _graphs, "in": in_buf, "out": out,
                           "start_pos": start_pos, "pool": _pool}
            self._graph_start_pos = start_pos
            print(f"[tinyinfer] decode CUDA graph captured at pos={start_pos} "
                  f"({len(_graphs)} chunks)", flush=True)
        except Exception as exc:
            # 捕获没成，把 _all_pinned 放回去：它不是性能开关，只对「路由已固定」成立
            for _ffn in locals().get("_pinned_on", []):
                try:
                    _ffn._all_pinned = False
                except Exception:
                    pass
            self._graph_failed = True
            self._graph = None
            # **捕获失败必须炸，不能静默降级。**
            # 以前这里只置 _graph_failed=True，然后 forward() 回落到 eager，
            # 于是「graph 从来没成功过」这件事只能靠翻代码发现 —— 实测在 L20
            # 上跑了整轮才看出来。一条路径要么成立，要么明确失败。
            import traceback as _tb
            _tb.print_exc()
            raise RuntimeError(
                f"decode CUDA graph capture failed ({exc!r}); there is no eager fallback"
            ) from exc


    def warm_graph_catalog(self) -> None:
        caps = list(getattr(self.tier, "graph_catalog", ()) or ())
        if not caps or self.device == "cpu":
            return
        ids = torch.tensor([[self.cfg.bos_token_id or 0]], device=self.device)
        for cap in caps:
            try:
                self._forward_impl(ids.repeat(1, min(int(cap), 8)), 0, None)
            except Exception:
                continue
            break

    def save_checkpoint(self, path) -> dict:
        import os as _os

        if not getattr(self.tier, "prefix_checkpoint", False):
            raise RuntimeError("prefix checkpoints are a tier-2 mechanism "
                               f"(current tier {self.tier.level.name})")
        snap = self.snapshot()
        tmp = str(path) + ".tmp"
        torch.save(snap, tmp)
        _os.replace(tmp, str(path))
        return {"path": str(path), "bytes": _os.path.getsize(str(path))}

    def load_checkpoint(self, path) -> None:
        if not getattr(self.tier, "prefix_checkpoint", False):
            raise RuntimeError("prefix checkpoints are a tier-2 mechanism "
                               f"(current tier {self.tier.level.name})")
        self.restore(torch.load(str(path), map_location=self.device, weights_only=False))

    def resample_from_hidden(self, hidden=None, temperature: float = 0.0):
        from .nn import rms_norm as _rms
        from .nn import sample_logits as _sample

        if not getattr(self.tier, "resume_sampling", False):
            raise RuntimeError("resume sampling is a tier-2 mechanism "
                               f"(current tier {self.tier.level.name})")
        h = self._last_hidden if hidden is None else hidden
        if h is None:
            raise RuntimeError("no hidden state to sample from")
        if h.dim() == 4:
            pm = getattr(self, "_last_pre_mix", None)
            if pm is not None and pm.shape[:2] == h.shape[:2]:
                y = (pm.unsqueeze(-1) * h.float()).sum(dim=2)
            else:
                y = h.float().mean(dim=2)
        else:
            y = h.float()
        y = _rms(y.to(self.out_norm.dtype), self.out_norm, self.eps)
        if self.head_on_cpu:
            logits = torch.nn.functional.linear(y.cpu().to(self.head.dtype), self.head).float().to(y.device)
        else:
            logits = torch.nn.functional.linear(y.to(self.head.dtype), self.head).float()
        return int(_sample(logits[0, -1], temperature))

    def forward(self, ids: torch.Tensor, start_pos: int = 0, chunk: int | None = None,
                positions=None):
        """Decode/prefill entry point: CUDA graph replay when captured, else the OOM ladder.

        positions: optional [b] per-row absolute position of each row's last token. A
        batched call mixing a fresh prompt with already-decoding sequences needs it --
        one shared start_pos would put every row at the same RoPE depth and the same
        compressed group. None (the default) is exactly the previous behaviour, and the
        CUDA-graph path only accepts the single-position form.
        """
        for attempt in range(3):
            try:
                # **图只覆盖「一次一个 token」的解码步。**
                #
                # prefill（ids.shape[1] > 1）走的是一批 token 的矩阵乘、窗口还没满、
                # kv_source 的压缩机也要凑够一整组才产出 index_k —— 这些都和图里
                # 固化的那份形状/地址不是一回事。以前这里没有按 seqlen 分流，prefill
                # 也会进来调 _graph_try_capture：它内部 _graph_capture_shape 直接
                # 返回 False，于是紧接着的 raise 把一次正常 prefill 判成致命错误。
                #
                # 实测：warmup 建好图之后，换一个 7 token 的 prompt 立刻抛
                #   RuntimeError: slot table ready but decode graph missing
                #   (ids=(1, 7) start_pos=0)
                # 也就是「同一进程里问第二句话就挂」。
                _graph_ok = ids.shape[1] == 1
                if start_pos >= 0 and chunk is None and _graph_ok:
                    hit = self._graph_run(ids, start_pos)
                    if hit is not None:
                        return hit
                    # **启动期：槽位表还没建，图不可能存在。**
                    #
                    # 捕获的前置条件之一是 device 侧的 [n_layers, n_experts] 槽位表，
                    # 而它由 pin_hot_experts() 建立；pin 又需要先跑过一次 forward 才
                    # 知道热点。于是「跑 forward 建表」和「建表才能捕获图」互为前提。
                    #
                    # 破法：**只有表还没建的那几次 forward 走直接路径**。这批调用是
                    # 启动期的探测（server.warmup 的两次 forward + pin），它们本来就不
                    # 是用户请求。表建好之后一律走 graph，没有任何回落。
                    if getattr(self, "_slot_tab_ready", False) is not True:
                        return self._forward_impl(ids, start_pos, chunk, positions=positions)
                    self._graph_try_capture(ids, start_pos)
                    hit = self._graph_run(ids, start_pos)
                    if hit is not None:
                        return hit
                    # 表已就绪却还没有图 = 真错误，不回落。
                    raise RuntimeError(
                        f"slot table ready but decode graph missing "
                        f"(ids={tuple(ids.shape)} start_pos={start_pos} "
                        f"gpu_layers={self.gpu_layers}/{self.max_layers})")
                return self._forward_impl(ids, start_pos, chunk, positions=positions)
            except Exception as exc:
                # torch raises OutOfMemoryError, while some CUDA paths surface a plain
                # RuntimeError whose text says "out of memory"; both are handled the same
                oom = isinstance(exc, torch.OutOfMemoryError) or "out of memory" in str(exc).lower()
                if attempt == 2 or not oom:
                    raise
                print(f"[tinyinfer] CUDA OOM: shedding VRAM (attempt {attempt + 1})", flush=True)
                self._shed_vram()
        raise RuntimeError("unreachable")

    def _forward_impl(self, ids: torch.Tensor, start_pos: int = 0, chunk: int | None = None,
                      positions=None):
        """Chunked prefill: KV/hash state carries across chunks (attention uses start_pos)."""
        if ids.device.type != self.device and self.device != "cpu":
            ids = ids.to(self.device)
        b, T = ids.shape
        # The VRAM expert-slot pool is disabled by the planner on this card: with slots on
        # the first decode step OOMs (3664MB base + 1.39GB ring + 1.2GB slots + the head
        # temporary is past the card), and pausing it for prefill does not help because
        # the allocation happens on the decode path anyway. Kept out of the hot path.
        # Single-chunk prefill is the numerically correct path: chunked prefill still
        # disagrees with it on the same positions (T-dependence in an early layer), which
        # showed up as garbage first tokens. Keep short prompts in one chunk; long ones
        # still step so an 8GB card cannot OOM.
        step = chunk or max(self.prefill_chunk, 32)
        # Measured now (64-token prompt): peak allocation is flat at 5.5GB for every chunk
        # from 64 to 512, and chunked prefill is bit-identical to single-chunk (maxdiff
        # 0.0), so the old 16//(T//16) bound was both stale and the reason long prompts
        # crawled one token at a time (~1.1 tok/s). A chunk that keeps the batch wide
        # keeps the SMs busy; 256 is the fastest measured point (3.5x chunk=8).
        if chunk is None and T > 256:
            step = max(1, min(step, 256))
        # 这一轮 forward 结束时告诉分级管理器「空闲了」：预热线程只在两次
        # forward 之间读盘，不和 decode 抢带宽。用 finally 保证异常路径也放行。
        _mgr = getattr(getattr(self, "expert_store", None), "_tier_mgr", None)
        try:
            if T <= step:
                return self._forward_chunk(ids, start_pos, positions=positions)
            logits = None
            for off in range(0, T, step):
                logits = self._forward_chunk(ids[:, off:off + step], start_pos + off)
                # the prefill borrows its expert reads from a private pool region; give
                # the whole region back between chunks (no DMA can be in flight here)
                _store = getattr(self, "expert_store", None)
                _reset = getattr(_store, "reset_prefill_pool", None)
                if _reset is not None:
                    _reset()
            return logits
        finally:
            if _mgr is not None:
                try:
                    _mgr.idle()
                except Exception:
                    pass

    def stream_generate(self, prompt_ids, max_new_tokens: int = 64, temperature: float = 0.0,
                        top_p: float = 1.0, top_k: int = 0, stop_ids=None, chunk: int | None = None):
        """Yield generated token ids one at a time (prefill once, then decode)."""
        from .nn import sample_logits

        ids = list(prompt_ids)
        self.reset()
        with torch.inference_mode():
            logits = self.forward(torch.tensor([ids], device=self.device), 0, chunk=chunk)
            if self.device != "cpu":
                torch.cuda.empty_cache()
            for step in range(max_new_tokens):
                nid = sample_logits(logits[0, -1], temperature, top_p, top_k)
                if stop_ids and nid in stop_ids:
                    break
                yield nid
                logits = self.forward(torch.tensor([[nid]], device=self.device),
                                      start_pos=len(ids) + step)

    # -- speculative decoding (MTP/DSpark) ---------------------------------
    def _window_span(self) -> int:
        """Tokens still replayable from the ring without cloning it."""
        spans = [getattr(b.attn, "window", 0) or 0 for b in self._blocks.values()]
        return min([s for s in spans if s] or [0]) or 0

    def _kv_truncate(self, target: int) -> None:
        """Roll every attention state back to `target` absolute tokens."""
        for b in self._blocks.values():
            a = b.attn
            win = getattr(a, "window", None) or 0
            keep = target if not win else min(target, win)
            kv = getattr(a, "_kv_window", None)
            if kv is not None:
                # 槽位 cache 不能切片：位置 p 的 KV 永远在 p % window 那一格里，
                # 回退只需把长度记回去并丢掉「窗口已满」的索引模板，重新生成时
                # 会覆盖同一批槽位。切 kv[:, :keep] 拿到的是槽位 0..keep，不是历史。
                if hasattr(a, "_kv_len"):
                    a._kv_len = keep
                a._decode_idxs = None
            if hasattr(a, "_decode_win_idxs"):
                a._decode_win_idxs = None
            comp = getattr(a, "compressor", None)
            if comp is not None and comp.ratio > 1:
                want = target % comp.ratio
                for st in comp._states.values():
                    pk = st.get("pkv")
                    if pk is not None and pk.shape[1] > want:
                        st["pkv"] = pk[:, :want].contiguous()
                        sc = st.get("psc")
                        if sc is not None:
                            st["psc"] = sc[:, :want].contiguous()

    def _draft(self, nid: int, pos: int, draft_len: int) -> list:
        """One DSpark draft block: target hiddens -> seed -> stages -> head."""
        stages = self.mtp
        s0 = stages[0]
        th = self._target_hidden
        if not th:
            return []
        keys = sorted(th.keys())
        if len(keys) < len(self.cfg.dspark_target_layer_ids):
            return []
        # a draft seeds its attention from the main KV of the current token; state from
        # the previous draft must not leak in (measured: draft 2 vs draft 3 disagreed
        # on the 4th token because the window kept the earlier block)
        for st in stages:
            st.attn.reset()
        main_x = s0.main_hiddens([th[k] for k in keys])
        tok = torch.tensor([[nid]], device=self.device)
        x = s0.forward_embed(tok)
        pre = torch.zeros(1, x.size(1), self.cfg.hc_mult, device=self.device)
        pre[:, :, 0] = 1.0
        for st in stages:
            x, pre = st.forward(x, pos, pre, main_x)
        dlogits, _conf = stages[-1].forward_head(x, pre, tok)
        if dlogits is None:
            return []
        import os as _os
        if _os.environ.get("TI_SPEC_DEBUG"):
            top = dlogits[0, 0].topk(3)
            print(f"[draft] pos={pos} main_x_mean={float(main_x.abs().mean()):.4f} "
                  f"hidden_mean={float(x.abs().mean()):.3f} top3={[int(i) for i in top.indices]}",
                  flush=True)
        n = min(draft_len, dlogits.size(1))
        return [int(dlogits[0, i].argmax()) for i in range(n)]
    def load_mtp(self) -> bool:
        """Load the three DSpark stages if their shards are on disk."""
        from .deepseek.mtp import load_mtp
        stages = load_mtp(self.store, self.cfg, quant=self.tier.keep_weights_quantized,
                          feedback_window=bool(getattr(self.tier, "mtp_feedback_window", True)))
        if stages is None:
            return False
        for st in stages:
            st.embed = getattr(self, "embed", None)  # drafts reuse the main embedding
            st.head = getattr(self, "head", None)    # drafts reuse the main output head
            st.to(self.device)
        self.mtp = stages
        return True

    def speculative_generate(self, prompt_ids, max_new_tokens: int = 32, draft_len: int | None = None,
                             temperature: float = 0.0, eos_token_id=None):
        """MTP speculative decoding: one main forward verifies a whole draft block.

        Draft (official DFlash contract, see mtp.py):
            target hiddens of layers 37/38/39 -> main_proj/main_norm -> main_x
            golden token -> noise-filled draft block -> stage 0..2 -> head logits
        Verify:
            one main forward over the drafts; preds[t] is the argmax at each position;
            accept the longest matching prefix, then truncate the KV to pos+accepted so
            the rejected positions never enter the cache. The logits of the last
            accepted position become the next step's logits -- nothing is replayed.
        """
        from .nn import sample_logits

        if getattr(self, "mtp", None) is None and not self.load_mtp():
            raise RuntimeError("MTP weights are not available (model-00044..46 missing?)")
        draft_len = draft_len or self.cfg.dspark_block_size
        ids = list(prompt_ids)
        out: list[int] = []
        acc = drafted = 0
        with torch.inference_mode():
            logits = self.forward(torch.tensor([ids], device=self.device), 0)
            if logits is None:
                # forward 的 3 次重试都失败时本该 raise（循环末尾就是 raise），
                # 真出现 None 时下游 logits[0,-1] 只会抛
                # 'NoneType' object is not subscriptable，把真正的失败点埋掉。
                # 实测 10 次请求里有 1 次会这样，先让它说清楚。
                raise RuntimeError(
                    "forward() returned None on prefill (engine state reset?)")
            if self.device != "cpu":
                torch.cuda.empty_cache()
            pos = len(ids)
            while len(out) < max_new_tokens:
                nid = int(sample_logits(logits[0, -1], temperature))
                out.append(nid)
                if eos_token_id is not None and nid == eos_token_id:
                    break
                if len(out) >= max_new_tokens:
                    break
                logits = self.forward(torch.tensor([[nid]], device=self.device), start_pos=pos)
                pos += 1
                drafts = self._draft(nid, pos, draft_len)
                if not drafts:
                    continue
                span = self._window_span()
                cheap = span == 0 or pos + len(drafts) <= span
                snap = None if cheap else self.snapshot()
                vlogits = self.forward(torch.tensor([drafts], device=self.device), start_pos=pos)
                preds = [logits[0, -1]] + [vlogits[0, t] for t in range(len(drafts) - 1)]
                j = 0
                for t, cand in enumerate(drafts):
                    if int(preds[t].argmax()) == cand:
                        j += 1
                    else:
                        break
                import os as _os
                if _os.environ.get("TI_SPEC_DEBUG"):
                    _p0 = int(preds[0].argmax())
                    print(f"[spec] pos={pos} draft0={drafts[0]} pred0={_p0} "
                          f"match={_p0 == drafts[0]} logits_argmax={int(logits[0, -1].argmax())} "
                          f"n_drafts={len(drafts)}", flush=True)
                drafted += len(drafts)
                acc += j
                if j < len(drafts):
                    if cheap:
                        self._kv_truncate(pos + j)
                    else:
                        self.restore(snap)
                        if j:
                            vlogits = self.forward(
                                torch.tensor([drafts[:j]], device=self.device), start_pos=pos)
                if j:
                    out.extend(drafts[:j])
                    pos += j
                    logits = vlogits[0, j - 1:j]
                    if eos_token_id is not None and drafts[j - 1] == eos_token_id:
                        break
        rate = acc / drafted if drafted else 0.0
        return out[:max_new_tokens], rate


    def generate(self, prompt_ids, max_new_tokens: int = 16, temperature: float = 0.0, eos_token_id=None,
                 top_p: float = 1.0, top_k: int = 0, start_pos: int = 0, keep_session: bool = False):
        """Greedy/sampled generation loop (prefill once, then decode step by step)."""
        # a generation starts from a clean session: leftover KV from a previous call
        # makes the first step read stale state (observed as a first-token failure)
        if not keep_session:
            self.reset()
        ids = list(prompt_ids)
        if eos_token_id is None:
            # stop at the checkpoint's own end-of-sentence token by default; without it a
            # greedy run kept emitting <|begin of sentence|> and a fresh "# ..." turn
            eos_token_id = getattr(getattr(self, "_tokenizer", None), "eos_token_id", None)
        with torch.inference_mode():
            logits = self.forward(torch.tensor([ids], device=self.device), 0)
            if logits is None:
                # forward 的 3 次重试都失败时本该 raise（循环末尾就是 raise），
                # 真出现 None 时下游 logits[0,-1] 只会抛
                # 'NoneType' object is not subscriptable，把真正的失败点埋掉。
                # 实测 10 次请求里有 1 次会这样，先让它说清楚。
                raise RuntimeError(
                    "forward() returned None on prefill (engine state reset?)")
            if self.device != "cpu":
                # the prefill kernels (tilelang workspace for the prompt-length shape)
                # leave ~5GB of cached blocks behind; against an 8GB card that pushes
                # reserved past physical memory and every decode step then pages
                # (measured 11498MB reserved, 0.10 tok/s). Hand them back to the driver.
                torch.cuda.empty_cache()
            out = []
            import time as _t
            _t0 = _t.perf_counter()
            for step in range(max_new_tokens):
                nid = int(sample_logits(logits[0, -1], temperature, top_p, top_k))
                out.append(nid)
                if eos_token_id is not None and nid == eos_token_id:
                    break
                logits = self.forward(torch.tensor([[nid]], device=self.device), start_pos=len(ids) + step)
                if __import__("os").environ.get("TI_TIME"):
                    _t1 = _t.perf_counter()
                    print(f"  decode {step} {_t1 - _t0:.2f}s", flush=True)
                    _t0 = _t1
        return out

    # -- introspection -----------------------------------------------------
    def stats(self) -> dict:
        gpu = sum(1 for i in self._blocks if self._layer_device(self._blocks[i]) != "cpu")
        out = {
            "layers_built": len(self._blocks),
            "layers_on_gpu": gpu,
            "max_layers": self.max_layers,
            "n_layers": self.cfg.n_layers,
            "gpu_layers_budget": self.gpu_layers,
            "tier": self.tier.level.name,
            "device": self.device,
        }
        if self.expert_store is not None:
            out["experts"] = self.expert_store.stats()
        if torch.cuda.is_available() and self.device != "cpu":
            out["vram_peak_mb"] = round(torch.cuda.max_memory_allocated() / 2 ** 20)
        return out
