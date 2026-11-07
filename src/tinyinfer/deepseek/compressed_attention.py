"""MLA attention with compressed KV (compress_ratio > 0), chunked-prefill capable.

Window KV uses a linear rolling history; compressed KV is a global group-indexed cache
shared from kv_source layers. Supports arbitrary (start_pos, seqlen) chunks and decode.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .compressor import Compressor, Indexer, shared_attn
from ..config import TinyConfig
from ..kernels.sparse_attn import sparse_attn_dispatch
from ..linear import QuantLinear
from ..nn import apply_rope, precompute_freqs_cis, rms_norm, window_idxs_general
from ..quant import dequant_fp8_block
from ..rope_cache import get_table, table_key
from ..weights import ShardStore


# kv_norm_rope_quant 的 launcher 缓存（几何固定：grid=1, block=128）
_KV_PREP: dict = {}
_BTGEMV: dict = {}

# _cs（kernel 要的 cos‖sin 打包向量）按 (表地址, 起点, 长度) 缓存。
# 一次 decode step 里所有压缩层拿到的都是同一张 freqs 表的同一行，原来每层
# 都重做一次 real/imag 的 CatArrayBatchedCopy（外加 3 个 host 侧 aten 调用）；
# 输入只有 (表, start_pos, s)，而 cat 是纯函数 —— 命中缓存与重算逐位相同。
_CS_CACHE: dict = {}


def _packed_cos_sin(table: torch.Tensor, start_pos: int, s: int) -> torch.Tensor:
    key = (str(table.device), int(table.data_ptr()), int(start_pos), int(s))
    hit = _CS_CACHE.get(key)
    if hit is None:
        if len(_CS_CACHE) > 64:
            _CS_CACHE.clear()
        freqs = table[start_pos:start_pos + s]
        hit = torch.cat([freqs.real, freqs.imag], dim=-1).reshape(-1).float().contiguous()
        _CS_CACHE[key] = hit
    return hit


class MLACompressedAttention:
    """MLA with compressed KV. Row-parallel split support mirrors MLAWindowAttention."""
    def __init__(self, store: ShardStore, layer_id: int, cfg: TinyConfig, *, quant: bool = False,
                 rt=None, tp=None, plan_fn=None):
        p = f"layers.{layer_id}.attn."
        self.plan_fn = plan_fn
        self.layer_id = layer_id
        self.tp = tp
        self.n_local_heads = cfg.n_heads
        self.rt = rt if rt is not None else shared_attn  # per-kv-source handoff slots
        self.cfg = cfg
        self.quant = quant
        self.eps = cfg.rms_norm_eps
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.rope_dim = cfg.qk_rope_head_dim
        self.o_groups = cfg.o_groups
        self.o_lora_rank = cfg.o_lora_rank
        self.window = cfg.sliding_window
        self.ratio = cfg.compress_ratios[layer_id]
        assert self.ratio > 0
        self.is_kv_source = layer_id in cfg.kv_source_layer_ids
        self.is_index_source = layer_id in cfg.index_source_layer_ids
        self.scale = self.head_dim ** -0.5

        if quant:
            self.wq_a = QuantLinear(store.get(p + "wq_a.weight"), store.get(p + "wq_a.scale"))
            self.wq_b = QuantLinear(store.get(p + "wq_b.weight"), store.get(p + "wq_b.scale"))
            self.wkv = QuantLinear(store.get(p + "wkv.weight"), store.get(p + "wkv.scale"))
            self.wo_a = QuantLinear(store.get(p + "wo_a.weight"), store.get(p + "wo_a.scale"))
            self.wo_b = QuantLinear(store.get(p + "wo_b.weight"), store.get(p + "wo_b.scale"))
            if plan_fn is not None:
                self._apply_plan(plan_fn)
        else:
            self.wq_a = dequant_fp8_block(store.get(p + "wq_a.weight"), store.get(p + "wq_a.scale"))
            self.wq_b = dequant_fp8_block(store.get(p + "wq_b.weight"), store.get(p + "wq_b.scale"))
            self.wkv = dequant_fp8_block(store.get(p + "wkv.weight"), store.get(p + "wkv.scale"))
            self.wo_a = dequant_fp8_block(store.get(p + "wo_a.weight"), store.get(p + "wo_a.scale"))
            self.wo_b = dequant_fp8_block(store.get(p + "wo_b.weight"), store.get(p + "wo_b.scale"))
        self.q_norm = store.get(p + "q_norm.weight")
        self.kv_norm = store.get(p + "kv_norm.weight")
        self.sink = store.get(p + "attn_sink").float()
        if tp is not None and tp.world > 1:
            self._shard_for_tp(tp)

        # DeepSeek's own definition (config.json -> rope_scaling): YaRN with factor 16
        # from the 65536-token training context, applied to the compressed-KV layers.
        # The reference model.py defaults these to 0/40 only because its flat ModelArgs
        # never reads the nested rope_scaling block.
        _rs = cfg.rope_scaling or {}
        from ..rope_cache import get_table, table_key

        _kargs = (self.rope_dim, min(cfg.max_position_embeddings, 1_048_576),
                  int(_rs.get("original_max_position_embeddings", 65536)),
                  cfg.compress_rope_theta, int(_rs.get("factor", 1)),
                  float(_rs.get("beta_fast", 32.0)), float(_rs.get("beta_slow", 1.0)))
        self._rope_key = table_key(*_kargs)
        self._rope_factory = (lambda _a=tuple(_kargs): precompute_freqs_cis(*_a))
        self.freqs_cis = get_table(self._rope_key, self._rope_factory)
        self.compressor = Compressor(store, layer_id, cfg, quant=quant, rt=rt) if self.is_kv_source else None
        self.indexer = Indexer(store, layer_id, cfg, self.freqs_cis, rt=rt) if self.is_index_source else None
        # per-layer runtime state (shared streaming slots serve several layers)
        self._states: dict[int, dict] = {}
        self._active_layer = 0

    def _apply_plan(self, plan_fn) -> None:
        """Split this layer's projections across cards (same rule as the window path).

        Every projection is row-parallel, so each card owns whole output rows and nothing
        has to be reduced. wq_a/wq_b/wkv/wo_b split on the output dimension; wo_a is
        handled by the caller because its layout is grouped (o_groups, o_lora, -1).
        """
        from ..tp import shard_quant_linear

        def split(name, ql):
            w = getattr(ql, "weight", None)
            if w is None or getattr(w, "dim", lambda: 0)() != 2:
                return ql
            plan = plan_fn(name, int(w.shape[0]))
            if not plan or len(plan) <= 1:
                return ql
            return shard_quant_linear(ql.weight, ql.scale, plan)

        self.wq_a = split("wq_a", self.wq_a)
        self.wq_b = split("wq_b", self.wq_b)
        self.wkv = split("wkv", self.wkv)
        self.wo_b = split("wo_b", self.wo_b)

    def _st(self) -> dict:
        st = self._states.get(self._active_layer)
        if st is None:
            st = self._states[self._active_layer] = {}
        return st

    @property
    def _kv_window(self):
        return self._st().get("kv")

    @_kv_window.setter
    def _kv_window(self, value):
        self._st()["kv"] = value

    @property
    def _kv_len(self) -> int:
        """已写入 cache 的绝对长度（环形槽位下不等于 buffer 宽度）。"""
        return int(self._st().get("klen", 0))

    @_kv_len.setter
    def _kv_len(self, value) -> None:
        self._st()["klen"] = int(value)

    @property
    def _kv_buf(self):
        """Pre-allocated sliding-window staging buffer (fixed address, no per-token cat)."""
        return self._st().get("winbuf")

    @_kv_buf.setter
    def _kv_buf(self, value):
        self._st()["winbuf"] = value

    @property
    def _compress_kv(self):
        return self._st().get("ckv")

    @_compress_kv.setter
    def _compress_kv(self, value):
        self._st()["ckv"] = value

    @property
    def _decode_win_idxs(self):
        return self._st().get("idxs")

    @_decode_win_idxs.setter
    def _decode_win_idxs(self, value):
        self._st()["idxs"] = value

    def reset(self) -> None:
        self._states.clear()
        if self.compressor is not None:
            self.compressor.reset()
        if self.indexer is not None:
            self.indexer.reset()
        self.rt.compress_kv = None
        self.rt.index_k = None
        self.rt.topk_idxs = None
        self.rt.candidates = None
        # 跨层复用的 per-step 中间量：换序列时必须丢掉，否则上一条序列的
        # positions/top-k 会被当成同一步的输入（见 _grp0_for/_latent_freqs/_idxs_all）。
        for _slot in ("_grp0_hit", "_fr_hit", "_idxs_hit", "_winrow_hit"):
            setattr(self.rt, _slot, None)

    def to(self, device):
        for name in ("wq_a", "wq_b", "wkv", "wo_a", "wo_b"):
            mod = getattr(self, name)
            if hasattr(mod, "to"):
                mod.to(device)
            else:
                setattr(self, name, mod.to(device))
        for name in ("q_norm", "kv_norm", "sink"):
            setattr(self, name, getattr(self, name).to(device))
        self.freqs_cis = self.freqs_cis.to(device)
        if self.compressor is not None:
            self.compressor.to(device)
        if self.indexer is not None:
            self.indexer.to(device)
        return self

    def _shard_for_tp(self, tp) -> None:
        """Same MLA split as MLAWindowAttention: q_lora and heads column-parallel,
        wo row-parallel with an all-reduce at the end; wkv replicated (one KV head)."""
        from ..tp import shard_linear

        if self.n_heads % tp.world:
            raise ValueError(f"n_heads {self.n_heads} not divisible by tp {tp.world}")
        self.n_local_heads = self.n_heads // tp.world
        self.wq_a = shard_linear(self.wq_a, tp, axis="out")
        lo, hi = tp.columns(self.q_norm.numel())
        self.q_norm = self.q_norm[lo:hi].contiguous()
        self.wq_b = shard_linear(self.wq_b, tp, axis="out")
        lo, hi = tp.columns(self.n_heads)
        self.sink = self.sink[lo:hi].contiguous()
        # wo_a is [o_groups * o_lora_rank, dim] and wo_b is [dim, o_groups * o_lora_rank]:
        # the o_lora_rank axis lives INSIDE each group, so both sides have to be split with
        # the group-major reshape (shard_grouped), not by a contiguous slice and not only
        # when the stored weight happens to be 3-D. Before this, wo_a was never split (the
        # stored weight is 2-D) while wo_b was, so any tp>1 run mismatched the grouped
        # matmul: wo_a produced groups*o_lora_rank columns and wo_b wanted a contiguous
        # fraction of them.
        from ..tp import shard_grouped

        w = getattr(self.wo_a, "weight", None)
        if w is not None:
            rlo, rhi = tp.rows(self.o_lora_rank)
            if isinstance(self.wo_a, torch.Tensor):
                gr, dim = w.shape
                self.wo_a = (w.view(self.o_groups, gr // self.o_groups, dim)[:, rlo:rhi]
                             .contiguous().view(-1, dim))
            else:
                self.wo_a = shard_grouped(self.wo_a, tp, self.o_groups, axis="out")
            self.o_lora_rank = rhi - rlo
        self.wo_b = shard_grouped(self.wo_b, tp, self.o_groups, axis="in")
        self.tp_world = tp.world
        self.tp_rank = tp.rank

    # ---- 跨层复用的 per-step 中间量 ----
    # 下面几个方法缓存的东西都是「同一次 forward 里所有压缩层共用的中间量」。
    # 槽位挂在 self.rt 上：chunk-major 执行时那是进程级单例，同一步的 38 个压缩层
    # 看到的是同一个对象；layer-major 时每个 kv_source 组一个实例。命中条件里都带着
    # 输入的身份/地址，所以命中时算的与重算的逐位相同（这些量都是纯函数）。

    def _grp0_for(self, positions: torch.Tensor, start_pos: int, s: int, b: int) -> torch.Tensor:
        """positions // ratio（每行的压缩组号）。原来每层重算一次整数除法。

        一个 forward 里所有层拿到的是同一个 positions 张量，且没有任何一层写它，
        所以同一 (张量, start_pos, s, b, ratio) 下重算与复用逐位相同。
        key 里带 numel/start_pos/s/b/ratio：换一步（哪怕张量被复用）就重建。
        """
        rt = self.rt
        key = (int(positions.data_ptr()), int(positions.numel()), int(start_pos), int(s),
               int(b), int(self.ratio))
        hit = getattr(rt, "_grp0_hit", None)
        if hit is not None and hit[0] == key and hit[1] is positions:
            return hit[2]
        out = (positions // self.ratio).to(torch.long)
        rt._grp0_hit = (key, positions, out)   # 持引用：地址在缓存期间不会被回收
        return out

    def _latent_freqs(self, grp0: torch.Tensor, ng: int, b: int, device) -> torch.Tensor:
        """latent 的 rope 表行 [b, ng, dim]。原来每层 arange+add+mul+gather 四个 kernel。

        输入只有 (grp0, ng, b, ratio, self.freqs_cis)，全是只读；gather 是纯函数，
        所以复用与重算逐位相同。grp0 本身也是 _grp0_for 的缓存对象，层间是同一个。
        """
        rt = self.rt
        key = (int(grp0.data_ptr()), int(grp0.numel()), int(ng), int(b), int(self.ratio))
        hit = getattr(rt, "_fr_hit", None)
        if hit is not None and hit[0] == key and hit[1] is grp0:
            return hit[2]
        grp = grp0.unsqueeze(1) + torch.arange(ng, device=device).unsqueeze(0)
        fr = self.freqs_cis[(grp * self.ratio).reshape(-1)].reshape(b, ng, -1)
        rt._fr_hit = (key, grp0, fr)
        return fr

    def _win_idxs_per_row(self, positions: torch.Tensor, s: int, n_old: int, device):
        """batch>1 时每行的窗口槽位模板。原来每层都在 Python 里逐行重建。

        window_idxs_per_row 对每一行跑 arange/-/>=/&/full_like/where/int 一长串
        kernel（一行 ~7 个 launch），而同一 forward 里 38 个压缩层拿到的是同一个
        positions、同一个 n_old —— 逐层重算是纯重复。输入全只读，函数是确定性的，
        命中缓存与重算逐位相同。
        """
        rt = self.rt
        key = (int(positions.data_ptr()), int(positions.numel()), int(s), int(n_old),
               int(self.window), str(device))
        hit = getattr(rt, "_winrow_hit", None)
        if hit is not None and hit[0] == key and hit[1] is positions:
            return hit[2]
        from ..nn import window_idxs_per_row
        out = window_idxs_per_row(positions, s, n_old, self.window, device)
        rt._winrow_hit = (key, positions, out)   # 持引用：地址在缓存期间不会被回收
        return out

    def _idxs_all(self, win_idxs: torch.Tensor, compress_idxs: torch.Tensor,
                  b: int, s: int) -> torch.Tensor:
        """[window 槽位 | 压缩槽位] 的拼接结果，按同一份输入复用。

        一个 decode step 里，窗口槽位模板是同一块存储的视图，top-k 在 30 个
        非 index_source 层之间是同一个 rt.topk_idxs 对象 —— 原来每层都把这两段
        各拼一次（早期 token 还要先 torch.full 一维 -1 再 cat 一次）。
        命中条件 = 同一 win 存储地址 + 同一 compress 张量对象 + 同形状，此时
        输入与重算完全一致，cat 是纯函数，所以输出逐位相同。
        缓存里同时持有 win 张量的引用：这样它的存储在整个缓存期间不会被回收，
        地址就不可能被另一个内容不同的张量复用（地址相同 ⇒ 同一块只读模板）。
        """
        rt = self.rt
        _k = int(self.cfg.index_topk)
        off = int(win_idxs.size(-1))
        t = int(compress_idxs.size(-1))
        if t > _k:                      # indexer 的 topk ≤ index_topk，这里只做兜底
            return torch.cat([win_idxs, compress_idxs], dim=-1)
        key = (int(win_idxs.data_ptr()), int(win_idxs.numel()), int(b), int(s), off, _k, t)
        hit = getattr(rt, "_idxs_hit", None)
        if hit is not None and hit[0] == key and hit[2] is compress_idxs:
            return hit[3]
        _win = win_idxs
        _src = compress_idxs
        if t < _k:
            # 右补到 index_topk：序列早期 t = min(index_topk, end_pos // ratio) 比
            # index_topk 小，而 layer-major 预填时两层会拿不同宽度（8 vs 7 列）把
            # cat 撑爆。-1 是 kernel 的空槽标记，补位不进入任何数值，只让 kernel
            # 的宽度恒定。
            _pad = torch.full((*compress_idxs.shape[:-1], _k - t), -1,
                              dtype=compress_idxs.dtype, device=compress_idxs.device)
            compress_idxs = torch.cat([compress_idxs, _pad], dim=-1)
        out = torch.cat([win_idxs, compress_idxs], dim=-1)
        # 记下未补位的原始 top-k 张量：下次按对象身份比较；_win 只是钉住存储地址
        rt._idxs_hit = (key, _win, _src, out)
        return out

    def forward(self, x: torch.Tensor, start_pos: int = 0,
                positions: torch.Tensor | None = None) -> torch.Tensor:
        # positions: optional [b] absolute position of each row's LAST token. A batched
        # call has rows at different depths (one sequence just prefilled, another already
        # decoding), and every per-row quantity below -- the RoPE row, the compressed
        # group index -- is a function of that row's own position, not of one shared
        # start_pos. None keeps the single-sequence path exactly as it was.
        import os as _os
        _pt = {} if _os.environ.get("TI_ATTN_PROF") else None
        _t0 = __import__("time").perf_counter() if _pt is not None else 0.0
        b, s, _ = x.shape
        if self.compressor is not None:
            self.compressor._active_layer = self._active_layer
        if self.indexer is not None:
            self.indexer._active_layer = self._active_layer
        # one shared table per (dim, seqlen, yarn args): resolving it through the cache
        # (rather than .to() per instance) keeps a single device copy instead of 40
        self.freqs_cis = get_table(self._rope_key, self._rope_factory, x.device)
        freqs = self.freqs_cis[start_pos:start_pos + s]

        def _lin(w, x_):
            return w.forward(x_) if hasattr(w, "forward") else F.linear(x_, w)

        # wq_a 和 wkv 吃的是同一个 h，量化一次给两个用。原来每个 fp8_linear 内部
        # 各跑一遍 act_quant（每层七次，每次一个 launch 加一趟全局往返），换来的
        # 却是同一份 fp8 激活。
        _shared_q = None
        if (hasattr(self.wq_a, "weight") and hasattr(self.wkv, "weight")
                and x.is_cuda and x.dtype == torch.bfloat16
                and int(x.shape[-1]) % 32 == 0):
            from ..kernels.quant import act_quant as _aq
            _xf = x.reshape(-1, int(x.shape[-1]))
            _shared_q = _aq(_xf, 32, "ue8m0", torch.float8_e8m0fnu)
        if _shared_q is not None:
            from ..gemm import fp8_linear_dual
            _aq_x, _asc_x = _shared_q
            # 两个投影一次 launch（vLLM 的 FusedA 做法：拼进同一个 grid）
            _qa, _kv_raw = fp8_linear_dual(
                _aq_x, _asc_x,
                self.wq_a.weight, self.wq_a.scale,
                self.wkv.weight, self.wkv.scale)
            qr = rms_norm(_qa.view(*x.shape[:-1], int(self.wq_a.weight.shape[0])),
                          self.q_norm, self.eps)
            q = _lin(self.wq_b, qr).view(b, s, self.n_local_heads, self.head_dim)
            apply_rope(q[..., -self.rope_dim:], freqs)
            _kv_raw = _kv_raw.view(*x.shape[:-1], int(self.wkv.weight.shape[0]))
        else:
            qr = rms_norm(_lin(self.wq_a, x), self.q_norm, self.eps)
            q = _lin(self.wq_b, qr).view(b, s, self.n_local_heads, self.head_dim)
            apply_rope(q[..., -self.rope_dim:], freqs)
            _kv_raw = _lin(self.wkv, x)
        # ---- KV 一段：norm + rope + fp8 roundtrip + 写槽位，一个 kernel ----
        # 原来是 rms_norm / apply_rope / act_quant(inplace) 三次 launch，加一次
        # buf[:, slot] = kv 的拷贝。kv_norm_rope_quant.cu 把这四步收成一个：
        # 它复现每一步的 bf16 取整（RoPE 写回 bf16、量化器读到的是取整后的行），
        # 所以数值与原来逐位相同，而 launch 少三次。
        kv = None
        _kv_prepped = False
        if (b == 1 and s == 1 and _kv_raw.is_cuda
                and _kv_raw.dtype == torch.bfloat16
                and int(_kv_raw.shape[-1]) == 512
                and int(self.head_dim) == 512 and int(self.rope_dim) == 64):
            _fb = self._kv_buf
            if (_fb is None or _fb.shape[0] != b or _fb.shape[1] != self.window
                    or _fb.dtype != _kv_raw.dtype or _fb.device != _kv_raw.device):
                _fb = _kv_raw.new_empty(b, self.window, self.head_dim)
                self._kv_buf = _fb
                self._decode_win_idxs = None
            # kernel 要的是 cos‖sin（前 rot/2 个 cos、后 rot/2 个 sin），不是交错的
            # (cos0,sin0,cos1,sin1,...)。给错了就是 rope 区 64 维里错 63 维。
            # 打包结果只由 (freqs 表, start_pos, s) 决定，而同一 decode step 里 38 个
            # 压缩层用的是同一张表、同一个位置 —— 原来每层各做一次 cat（一个
            # CatArrayBatchedCopy launch 加几次 host 侧 aten 调度），现在一次 step 一次。
            # 数值不变：cat 是纯函数，命中缓存返回的就是同样那 64 个数。
            _cs = _packed_cos_sin(self.freqs_cis, start_pos, s)
            _kvfn = _KV_PREP.get("fn")
            if _kvfn is None:
                from ..kernels import native as _nat
                _kvfn = _KV_PREP["fn"] = _nat.BoundLaunch(
                    _nat.kernel("kv_norm_rope_quant", "kv_norm_rope_quant"),
                    (1,), (128,), "TTTTif")
            _kvfn(_kv_raw.reshape(-1), self.kv_norm.reshape(-1), _cs,
                  _fb.reshape(-1), int(start_pos) % self.window,
                  float(self.eps), stream=None)
            kv = _fb
            _kv_prepped = True
        if kv is None:
            kv = rms_norm(_kv_raw, self.kv_norm, self.eps)
            apply_rope(kv[..., -self.rope_dim:], freqs)
            # 参考实现在 RoPE 之后把窗口 KV 过一遍 fp8 再回来，cache 里存的就是那个
            # 形式；跳过 round trip 是 ~4.7% 的输出差异，放在旋转之前又是另一个值。
            if getattr(self, "quant", False) and kv.is_cuda:
                from ..kernels.quant import act_quant
                kv = act_quant(kv, 32, "ue8m0", torch.float8_e8m0fnu, True)


        # ---- vLLM 式槽位 cache（同 MLAWindowAttention）----
        # 位置 p 写进 p % window 那一格，历史永不移动。旧实现在这里做三份拷贝：
        # 历史左移（clone + copy_）、新块 copy_、以及把不连续切片交给 attention
        # 时的 .contiguous() —— 每层每 token 都在搬整个窗口。
        buf = self._kv_buf
        if (buf is None or buf.shape[0] != b or buf.shape[1] != self.window
                or buf.dtype != kv.dtype or buf.device != kv.device):
            buf = kv.new_empty(b, self.window, self.head_dim)
            self._kv_buf = buf
            self._decode_win_idxs = None
        if _kv_prepped:
            pass                       # kv_norm_rope_quant 已经写进槽位
        elif s == 1:
            buf[:, start_pos % self.window] = kv[:, 0]
        else:
            if s > self.window:
                _kv = kv[:, s - self.window:]
                _slots = torch.arange(start_pos + s - self.window, start_pos + s,
                                      device=x.device) % self.window
            else:
                _kv = kv
                _slots = torch.arange(start_pos, start_pos + s, device=x.device) % self.window
            buf.index_copy_(1, _slots, _kv)
        kv_w = buf
        n_old = min(start_pos, self.window)
        full = s == 1 and n_old == self.window
        if full and self._decode_win_idxs is not None:
            win_idxs = self._decode_win_idxs
        else:
            if positions is not None and positions.numel() == b and b > 1:
                # 同一步里所有层算的是同一份每行窗位，改走跨层缓存（见 _win_idxs_per_row）
                win_idxs = self._win_idxs_per_row(positions, s, n_old, x.device)
            else:
                from ..nn import window_idxs_ring_cached
                win_idxs = window_idxs_ring_cached(start_pos, s, self.window,
                                                   x.device).unsqueeze(0).expand(b, -1, -1)
            if full:
                self._decode_win_idxs = win_idxs
        # 整块 cache 就是 attention 的输入：地址固定、连续，没有切片也没有拷贝
        self._kv_window = buf
        self._kv_len = start_pos + s

        # ---- compressed KV ----
        end_pos = start_pos + s
        ratio_now = self.ratio if self.is_kv_source else (self.rt.src_ratio or self.ratio)
        compress_len = end_pos // ratio_now
        if _pt is not None:
            _t1 = __import__("time").perf_counter()
        latent = self.compressor.forward(x, start_pos) if self.compressor is not None else None
        if _pt is not None:
            _pt["compressor"] = (__import__("time").perf_counter() - _t1) * 1e3
        # Per-row group index, resolved BEFORE the RoPE block below needs it. With one
        # shared start the whole batch lands at the same group offset, so two sequences
        # at different depths overwrite each other's compressed history -- measured as
        # row A drifting 16.18 against the same prompt run alone.
        if positions is not None and positions.numel() == b:
            # 每行压缩组号。同一 forward 里所有层看到的是同一个 positions 张量、
            # 且没有一层写它，所以这一格除法可以跨层复用（见 _grp0_for）。
            # 数值不变：同一输入、同一整数除法，结果逐位相同。
            _grp0 = self._grp0_for(positions, start_pos, s, b)
        else:
            _grp0 = None
        if latent is not None:
            # the reference rotates the latent at its group position and quantises it to fp4
            # (block 16, E4M3 scales) before it enters the shared cache; both steps are part
            # of the stored value, not an optimisation
            if getattr(self, "quant", False) and latent.is_cuda:
                from ..nn import apply_rope as _apply_rope
                from ..kernels.fp4q import fp4_act_quant
                # One RoPE row per latent group; group j rotates at its window's first
                # token position. With per-row positions the rows differ, so this becomes
                # a gather instead of an arange (a batched call can hold one sequence at
                # group 0 and another at group 40).
                _ng = int(latent.size(1))
                if _grp0 is not None:
                    # 原来这里是 arange(1) + add(1) + mul(1) + index_select(1) 四个
                    # kernel，每层都算一遍；输入 (grp0, ng, b, ratio, 表) 在同一步里
                    # 对所有层相同，改走 _latent_freqs 的单槽缓存。数值不变：
                    # gather 是纯函数，命中时拿到的行与重算的完全相同。
                    fr = self._latent_freqs(_grp0, _ng, b, latent.device)
                elif start_pos == 0:
                    fr = self.freqs_cis[:_ng * self.ratio : self.ratio]
                else:
                    _g0 = start_pos // self.ratio
                    _grp = torch.arange(_g0, _g0 + _ng, device=latent.device)
                    fr = self.freqs_cis[_grp * self.ratio]
                # 旋转+fp4 往返直接做在 latent 上，不再 clone。数值原因：这次 forward
                # 里 latent 在旋转之后只被当作「旋转+量化后的结果」使用（写共享 cache、
                # 交给 indexer —— 索引器拿到的本来就是旋转后的张量），旋转前的值没有
                # 读者。clone 只是一次逐字节的 DtoD 拷贝，原张量上做的原地旋转/量化
                # 与在副本上做的完全同一组运算，得到的字节也完全相同，所以省掉这次
                # 拷贝不改变任何数值。
                if fr.dim() == 3:
                    # per-row table [b, ng, d]: rotate each row on its own, because a
                    # single [ng, d] table cannot describe two rows at different depths
                    for _r in range(latent.size(0)):
                        _apply_rope(latent[_r:_r + 1, ..., -self.rope_dim:],
                                    fr[_r:_r + 1].reshape(fr.size(1), -1))
                else:
                    _apply_rope(latent[..., -self.rope_dim:], fr)
                fp4_act_quant(latent, 16, True, scale_dtype=torch.float8_e4m3fn)
            # Per-row group index. With a shared start the batch would store every row at
            # the same group offset, so two sequences of different length overwrite each
            # other's compressed history -- that is what made the second one degrade the
            # first (measured: row A drifted by 16.18 against the same prompt alone).
            # 单行时不需要问 GPU：组索引就是 position // ratio，而 start_pos 已经是
            # Python int。原来的 int(_grp0.min().item()) 每层都要一次 DtoH 同步，
            # 40 层就是每 token 40 次 —— 每次同步都把 GPU 钉在原地等 CPU 回话，
            # 实测整个 decode 有 360 次 DtoH、GPU 利用率掉到 20-40%。
            # 只有 batch>1（每行位置不同）时才真的需要读回 GPU 的值。
            if _grp0 is None:
                start_group = start_pos // self.ratio
            elif _grp0.numel() == 1:
                start_group = start_pos // self.ratio
            else:
                start_group = int(_grp0.min().item())
            need = start_group + latent.size(1)
            cur = self._compress_kv
            # The cache is shared across the kv_source group via self.rt, so it can already
            # exist with a different batch width than this call wants (a single-sequence
            # warm-up, then a batched request). Reallocate on a width change instead of
            # letting the slice assignment broadcast-fail; the content of the old rows is
            # still valid, so carry it over when the old width fits.
            if cur is None or cur.shape[0] != b or cur.shape[1] < need:
                # pre-allocate a decode-sized run of groups so the buffer address stays
                # fixed for the lifetime of the conversation (CUDA graph capture needs
                # it); a long prompt still grows once, outside the captured step.
                cap = max(need, int(self.cfg.max_position_embeddings // max(1, self.ratio)))
                cap = min(cap, max(need, 4096))
                grown = torch.zeros(b, cap, self.head_dim,
                                    dtype=latent.dtype, device=latent.device)
                if cur is not None:
                    keep = min(cur.shape[1], cap)
                    grown[:, :keep] = cur[:, :keep].expand(b, -1, -1)[:, :keep]
                self._compress_kv = grown
            self._compress_kv[:, start_group : start_group + latent.size(1)] = latent
            self.rt.compress_kv = self._compress_kv
        if compress_len <= 0:
            compress_kv = compress_idxs = None
        elif self.rt.compress_kv is None:
            # 本批里有序列的 position 已经过了压缩阈值，但这个 kv_source 组还没
            # 写过任何压缩组（典型成因：批里混了刚 prefill 的序列和已经解码很久
            # 的序列，start_pos 取的是最小值，而判断走的是最大值一侧）。
            # 以前这里直接 self.rt.compress_kv[:, ...] 抛
            # TypeError: 'NoneType' object is not subscriptable，整批请求 500。
            # 退化成「本步不带压缩历史」，至少不炸整批。
            compress_kv = compress_idxs = None
        else:
            compress_kv = self.rt.compress_kv[:, :compress_len]
            if self.indexer is not None:
                if _pt is not None:
                    _t2 = __import__("time").perf_counter()
                compress_idxs = self.indexer.forward(x, qr, latent, start_pos, kv_w.size(1))
                if _pt is not None:
                    _pt["indexer"] = (__import__("time").perf_counter() - _t2) * 1e3
                self.rt.topk_idxs = compress_idxs
            else:
                compress_idxs = self.rt.topk_idxs

        if __import__("os").environ.get("TI_DEBUG_ATTN") and \
                getattr(self, "layer_id", -1) == 20:
            _ck = self.rt.compress_kv
            _ik = getattr(self.rt, "index_k", None)
            _cd = getattr(self.rt, "candidates", None)
            print(f"[attn20] start={start_pos} s={s} cl={compress_len} "
                  f"kv={None if compress_kv is None else tuple(compress_kv.shape)}"
                  f"|{0.0 if _ck is None else float(_ck.float().abs().sum()):.6f} "
                  f"ik={None if _ik is None else tuple(_ik.shape)}"
                  f"|{0.0 if _ik is None else float(_ik.float().abs().sum()):.6f} "
                  f"cand={None if _cd is None else float(_cd.sum()):.1f}", flush=True)
        if compress_idxs is not None and compress_idxs.size(-1) > 0:
            # kv_all 的 cat 留在这里：window 缓冲是每层自己的环形 cache（内容逐层不同），
            # 而 compress 段是跨层共享那一块的切片（内存上不挨着本层的 window），
            # kernel 又要求两段拼成一段连续的下标空间 —— 只能靠这一次拷贝。
            # 换成「预分配 + 两次 slice 写入」反而变成两个 launch（窗口一次、压缩一次），
            # 数值一样但 launch 更多，所以不做。
            kv_all = torch.cat([kv_w, compress_kv], dim=1)
            # 下标拼接（含早期 token 的 -1 右补）按「同一份输入」跨层复用，见 _idxs_all。
            # 同一 decode step 里窗口模板是同一块存储的视图、top-k 在 30 个非
            # index_source 层之间还是同一个 rt.topk_idxs 对象 —— 原来这 38 层各自
            # 重新算一遍 cat（早期 token 还要加一次 torch.full + cat），现在最多
            # 每个 index_source 一次。数值不变：cat 和补 -1 都是纯函数，
            # 命中时输入与重算完全一致，-1 依然只走 kernel 的空槽分支。
            idxs_all = self._idxs_all(win_idxs, compress_idxs, b, s)
        else:
            kv_all, idxs_all = kv_w, win_idxs

        if _pt is not None:
            _t3 = __import__("time").perf_counter()
        o = sparse_attn_dispatch(q, kv_all, self.sink, idxs_all, self.scale)
        if _pt is not None:
            _pt["sparse"] = (__import__("time").perf_counter() - _t3) * 1e3
            _pt["total"] = (__import__("time").perf_counter() - _t0) * 1e3
            print(f"[attn-prof] {_pt}", flush=True)
        apply_rope(o[..., -self.rope_dim:], freqs, inverse=True)

        o = o.view(b, s, self.o_groups, -1)
        if hasattr(self.wo_a, "weight"):  # fp8 resident: dequant just this grouped GEMM
            # 同 MLAWindowAttention：权重在一次 decode 里不变，反量化结果按层缓存。
            wo_a = self._st().get("wo_a_bf")
            if wo_a is None:
                from ..quant import dequant_fp8_block
                wo_a = dequant_fp8_block(self.wo_a.weight, self.wo_a.scale)
                self._st()["wo_a_bf"] = wo_a
        else:
            wo_a = self.wo_a
        wo_a = wo_a.view(self.o_groups, self.o_lora_rank, -1)
        # the official Attention quantises o to fp8 (and dequantises it back) before the
        # grouped einsum; skipping the round trip is a ~4.8% output difference, so it is
        # part of the reference semantics rather than an optimisation
        if getattr(self, "quant", False) and o.is_cuda:
            from ..kernels.quant import act_quant
            o = act_quant(o, 32, "ue8m0", torch.float8_e8m0fnu, True)
        # 这个 einsum 是「每个位置一个独立的 [r,d] x [d]」，走 cuBLAS 的 grouped
        # batched gemm，实测 3.65 ms/token（每层一个 91us 的 kernel，占 GPU 时间 15%）。
        # bt_gemv 把它铺成一个 grid：一个 warp 一行，g 在 blockIdx.y。
        if __import__("os").environ.get("TI_GPUDBG") == "1":
            print(f"[btgemv] cuda={o.is_cuda} shp={tuple(o.shape)} "
                  f"wdt={wo_a.dtype} wshp={tuple(wo_a.shape)}", flush=True)
        if (o.is_cuda and o.shape[0] * o.shape[1] == 1
                and wo_a.dtype == torch.bfloat16):
            _g, _r, _d = int(wo_a.shape[0]), int(wo_a.shape[1]), int(wo_a.shape[2])
            try:
                from ..kernels import native as _natb
                _xb = o.reshape(_g, _d).contiguous()
                _yb = torch.empty(_g, _r, dtype=torch.bfloat16, device=o.device)
                _key = ("bt_gemv", _g, _r, _d)
                _fn = _BTGEMV.get(_key)
                if _fn is None:
                    _fn = _natb.BoundLaunch(
                        _natb.kernel("bt_gemv", "bt_gemv_bf16"),
                        ((_r + 7) // 8, _g), (256,), "TTTOiii")
                    _BTGEMV[_key] = _fn
                _fn(_xb.view(-1), wo_a.reshape(-1).contiguous(), _yb.view(-1), _g, _r, _d)
                o = _yb.view(1, 1, _g, _r)
            except Exception as _exc:
                if __import__("os").environ.get("TI_GPUDBG") == "1":
                    import traceback as _tb
                    print(f"[btgemv-fail] {_exc!r}", flush=True)
                    _tb.print_exc()
                o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
        else:
            o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
        out = _lin(self.wo_b, o.flatten(2))
        if getattr(self, "tp_world", 1) > 1:
            from ..tp import all_reduce_sum

            out = all_reduce_sum(out)
        return out
