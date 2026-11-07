"""MoE with fp4 routed experts and one fp8 shared expert, loaded straight from HF shards.

Reference implementations (consult when fixing this file):
  * DeepInfer  deepinfer/model_executor/models/v4_modules/mlp.py  (DeepseekMoE:
        sqrtsoftplus gate scoring, n_activated_experts=6, route_scale, shared expert)
  * Official   reference/official/model.py (Gate / Expert FFN) and kernel.py
        (fused_fp4_expert_ffn: gate+up+silu / down, block-scaled fp4)
  * flash-moe  expert fetch/cache/eviction model (384-expert top-6 layout)

Checkpoint tensor names (do not rename):
  layers.{L}.ffn.gate.{weight,bias,bias_vl}
  layers.{L}.ffn.experts.{i}.{w1,w2,w3}.{weight,scale}    fp4 E2M1 + ue8m0
  layers.{L}.ffn.shared_experts.{w1,w2,w3}.{weight,scale} fp8 E4M3
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ..config import TinyConfig
from ..quant import dequant_fp4_block, dequant_fp8_block
from ..weights import ShardStore

# Diagnostic switch for expert parallelism, read once: the per-layer printout below is
# useful while EP is being finished but must cost nothing on a normal run (a getenv per
# forward would be a syscall per layer).
_EP_DBG = __import__("os").environ.get("TI_EP_DBG") == "1"

# Bound launchers for swiglu_scale, keyed by element count (the grid depends on it).
_SWIGLU_BOUND: dict = {}


def _swiglu_bound(n: int):
    from ..kernels import native

    hit = _SWIGLU_BOUND.get(n)
    if hit is not None:
        return hit
    bound = native.bind("swiglu_scale", "swiglu_scale", ((n + 255) // 256,), (256,),
                        "TTnTOif")
    _SWIGLU_BOUND[n] = bound
    return bound


class Gate:
    def __init__(self, store: ShardStore, layer_id: int, cfg: TinyConfig, topk: int = 0):
        p = f"layers.{layer_id}.ffn.gate."
        self.weight = store.get(p + "weight").float()
        self.bias = store.get(p + "bias").float()
        self.bias_vl = store.get(p + "bias_vl").float() if cfg.vision_enabled else None
        # 0 = checkpoint top-k; aggressive tiers pass a smaller k (fewer expert reads,
        # same fp4 format — quality traded, generation still coherent).
        self.topk = int(topk) if topk and 0 < topk < cfg.n_activated_experts else cfg.n_activated_experts
        self.score_func = cfg.scoring_func
        self.norm_topk_prob = cfg.norm_topk_prob
        self.route_scale = cfg.route_scale

    def forward(self, x: torch.Tensor, image_mask: torch.Tensor | None = None):
        # Host-side routing was tried here and reverted: computing the gate on the CPU is
        # arithmetically correct (indices and weights match the GPU path exactly on real
        # activations) and 0.29ms/layer, but it has to bring the activation down from the
        # GPU first, and that D2H sync costs more than the GPU matmul it replaces.
        # Measured: 788ms -> 879ms per token.
        logits = F.linear(x.float(), self.weight)  # gate_temp == 1.0 in the released config
        # vLLM's router does softplus(x).sqrt() + bias + top-k + renormalise + routing
        # scale in one kernel. The vision path is excluded: it picks a different bias per
        # token, and the kernel takes a single e_score_correction_bias for all tokens.
        # bias_vl only matters when a token is an image token, which is exactly what
        # image_mask encodes; with image_mask=None the torch path below also ignores it
        # and uses self.bias. Requiring "bias_vl is None" excluded every text-only run of
        # a vision-capable checkpoint, so this model never reached the fused router.
        if (self.score_func == "sqrtsoftplus" and image_mask is None):
            from .vllm_ops import gate_topk

            fused = gate_topk(logits, self.topk, self.norm_topk_prob,
                              self.route_scale, self.bias)
            if fused is not None:
                return fused[0], fused[1].long()
        scores = logits
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1)
        elif self.score_func == "sigmoid":
            scores = scores.sigmoid()
        else:
            scores = F.softplus(scores).sqrt()
        bias = self.bias
        if image_mask is not None and self.bias_vl is not None:
            bias = torch.where(image_mask.unsqueeze(-1), self.bias_vl, bias)
        # deterministic top-k: torch.topk's tie order is not stable on CUDA, so the same
        # layer could pick a different expert set on every call (measured: two forwards of
        # an identical input went to disjoint expert sets and the MoE output differed by
        # 0.49). A stable descending argsort always breaks ties by expert index.
        logits = scores + bias
        order = torch.argsort(logits, dim=-1, descending=True, stable=True)
        indices = order[..., :self.topk]
        weights = scores.gather(1, indices)
        if self.norm_topk_prob and self.topk > 1:
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        weights = weights * self.route_scale
        return weights, indices


# Set by Engine.forward: during prefill the expert cache must not admit (it would
# evict the decode working set and compete with the pinned layer slab for host RAM).
_admit_cache = True


_SWIGLU2 = None


def _swiglu2(gate: torch.Tensor, up: torch.Tensor, limit: float) -> torch.Tensor:
    """silu(clamp(gate,max)) * clamp(up,±limit)，一个 kernel。

    QuantLinear 的输出已经是 bf16，原来那串 .float() 只是换 dtype，不改数值；
    clamp×2 + silu + mul + 中间的 to() 一共七个算子，这里一次算完。
    """
    global _SWIGLU2
    if (not gate.is_cuda or gate.dtype != torch.bfloat16
            or up.dtype != torch.bfloat16 or up.shape != gate.shape):
        raise RuntimeError(
            "shared expert swiglu 需要同形状的 bf16 输入："
            f"gate={gate.dtype}{tuple(gate.shape)} up={up.dtype}{tuple(up.shape)}")
    from ..kernels import native
    if _SWIGLU2 is None:
        _SWIGLU2 = native.kernel("swiglu2", "swiglu2_bf")
    g = gate.reshape(-1).contiguous()
    u = up.reshape(-1).contiguous()
    n = int(g.numel())
    out = torch.empty(n, dtype=torch.bfloat16, device=gate.device)
    native.launch(_SWIGLU2, ((n + 255) // 256,), (256,),
                  [g, u, out, n, float(limit)])
    return out.view(gate.shape)


_MOE_FUSED: dict = {}


class Expert:
    """One SwiGLU FFN; routed experts are fp4, the shared expert is fp8."""

    def __init__(self, w1, w2, w3, swiglu_limit: float):
        self.w1, self.w2, self.w3 = w1, w2, w3
        self.swiglu_limit = swiglu_limit
        self.fp4 = None  # optional (w1,s1,w2,s2,w3,s3) tuple: compute without bf16 dequant
        # merged gate|up matrix: one GEMV produces both halves so vLLM's
        # silu_and_mul_with_clamp can consume them directly
        self.fp4_13 = None
        # Stable identity for the marlin repack cache: (layer, expert) or None for the
        # shared expert. An expert's fp4 tensors are views into the recycled _blob_stage
        # ring, so their addresses rotate and a data_ptr-keyed cache never hits -- measured
        # 2217 repack launches per decode step where ~240 would do, each costing 733 us
        # against 32.5 us for a hit. Carrying the identity down the call chain is the fix.
        self.key = None

    def to(self, device):
        # torch.Tensor.to returns a NEW tensor; dropping the result left the weights on
        # the CPU (crash: "mat2 is on cpu ... other tensors on cuda")
        for name in ("w1", "w2", "w3"):
            w = getattr(self, name)
            if w is None:
                continue
            if isinstance(w, torch.Tensor):
                setattr(self, name, w.to(device))
            elif hasattr(w, "to"):
                w.to(device)  # QuantLinear.to is in-place
        if self.fp4 is not None:
            self.fp4 = tuple(t.to(device) for t in self.fp4)
        return self

    def forward_fp4(self, x: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
        """fp4 kernel path; M==1 (decode) uses the GEMV kernels (GEMM wastes 31/32 rows)."""
        from ..gemm import fp4_linear

        _key = self.key
        w1, s1, w2, s2, w3, s3 = self.fp4
        # the fused prefill kernel is built for chunk-sized M; at M=2..8 it measured
        # 7.5x slower per layer than the GEMV path (667ms vs 89ms), so only real
        # prefill shapes use it
        if x.size(0) > 8 and x.is_cuda:
            # chunked prefill: one fused kernel does gate+up+silu, another does down+scale
            try:
                from .moe_fused import fused_fp4_expert_ffn
                w = None if weights is None else weights.reshape(-1)
                return fused_fp4_expert_ffn(x, w1, s1, w3, s3, w2, s2, w,
                                            limit=float(self.swiglu_limit))
            except Exception:
                pass
        gate = fp4_linear(x, w1, s1, key=_key)
        up = fp4_linear(x, w3, s3, key=_key)
        if self.swiglu_limit > 0:
            # Merge the two halves on the ACTIVATION (64KB at T=1), never on the
            # weights: stacking W1/W3 would cost another 36MB per expert per step.
            # vLLM's fused clamp+SwiGLU replaces clamp(clamp(silu(mul))) -- four
            # dispatches become one.
            from .vllm_ops import clamp_swiglu

            fused = clamp_swiglu(torch.cat((gate, up), dim=-1), self.swiglu_limit)
            if fused is not None:
                y = fused.float()
            else:
                gf, uf = gate.float(), up.float()
                uf = torch.clamp(uf, min=-self.swiglu_limit, max=self.swiglu_limit)
                gf = torch.clamp(gf, max=self.swiglu_limit)
                y = F.silu(gf) * uf
        else:
            y = F.silu(gate.float()) * up.float()
        if weights is not None:
            y = weights * y
        return fp4_linear(y.to(torch.bfloat16), w2, s2, key=_key)

    def forward_fp4_pq(self, aq: torch.Tensor, asc: torch.Tensor,
                       weights: torch.Tensor | None = None) -> torch.Tensor:
        """fp4 path with an activation quantised once for the whole expert batch.

        The quantiser does not depend on the expert, so serving six experts repeated it
        six times (48 rows at prefill); it is now computed once per layer.
        """
        # Our own w4a8 kernel. gemv_fp4 (the tilelang one) is no longer imported here at
        # all: its module-level tilelang import was the thing that failed on a deployment
        # without tilelang, and its kernel computes a different quantisation scheme
        # (w4a16) from the one the checkpoints define.
        from .fp4_native import gemv_fp4_pq

        gemv_fp4_m8 = None

        # Several tokens of one chunk can route to the same expert: the single-row GEMV
        # flattens a [rows, K] activation into the kernel's [K] slot and reads it wrong,
        # which is what made a multi-token forward disagree with token-by-token decoding.
        def _lin(aqq, assc, w, s, ws=None):
            # The multi-row kernel is exact on synthetic weights (M=1..8, rel 0) but
            # measured 42% off with real expert-34 weights, while the single-row kernel
            # is exact on both. Rows are therefore done one at a time unless the batch
            # kernel is explicitly requested; a chunk routes 1-2 rows per expert anyway.
            if (aqq.dim() == 2 and aqq.size(0) > 1
                    and __import__("os").environ.get("TI_MULTIROW_GEMV") == "1"):
                return gemv_fp4_m8(aqq, assc, w, s)
            if aqq.dim() == 2 and aqq.size(0) > 1:
                rows = [gemv_fp4_pq(aqq[i], assc[i], w, s, ws) for i in range(aqq.size(0))]
                return torch.cat(rows, 0)
            a1 = aqq[0] if aqq.dim() == 2 else aqq
            s1 = assc[0] if assc.dim() == 2 else assc
            return gemv_fp4_pq(a1, s1, w, s, ws)

        w1, s1, w2, s2, w3, s3 = self.fp4
        ws = (None, None, None)
        # NOTE: stacking [w1; w3] into one 2N-row GEMV was tried and removed: the two
        # torch.cat calls it needs cost 768 launches and 22ms of GPU time per token, more
        # than the 18 -> 12 launch saving. The GEMVs are not the bottleneck.
        # ws stays None. It used to hold a cached pow(2, s-127) expansion of the three
        # scale tables so the fp32-scale kernel could take them directly. The e8m0 kernel
        # now reads the checkpoint's bytes, which removes that conversion (measured 9.3 us
        # per expert matrix) and lets the decode path take the fast branch inside
        # gemv_fp4_pq instead of the fp32 one.
        # Two calls, not the fused gate_up kernel. The fusion was worth 1.50x when both
        # paths were the fp32-scale GEMV (it reads the activation once instead of twice),
        # but the e8m0 kernel stages the activation in shared memory anyway, and the
        # fused wrapper still takes fp32 scales. Measured on [2304, 5120] x 2:
        #     fused gate_up   250.7 us   (and it needs a pow(2, s-127) pass first)
        #     two e8m0 calls   42.2 us
        gate_t = _lin(aq, asc, w1, s1, ws[0])
        up_t = _lin(aq, asc, w3, s3, ws[2])
        # One launch for clamp + silu + multiply + routing weight. The torch spelling was
        # seven kernels per layer (two dtype copies, two clamps, silu, mul, mul) and the
        # tensors are [1, 2304] -- so the cost was the launches, not the bytes.
        _n = int(gate_t.numel())
        y = torch.empty_like(gate_t)
        _bound = _swiglu_bound(_n)
        _bound(gate_t.reshape(-1), up_t.reshape(-1),
               None if weights is None else weights.reshape(-1),
               y.reshape(-1), _n, float(self.swiglu_limit), stream=None)
        from ..gemm import fp4_linear
        return fp4_linear(y, w2, s2)

    def _scratch(self, nbytes: int, device):
        """MoE 融合 kernel 的中间 buffer（fp8 激活 + e8m0 scale），按大小复用。"""
        buf = getattr(self, "_scratch_buf", None)
        if buf is None or buf.numel() < nbytes or buf.device != device:
            buf = torch.empty(nbytes, dtype=torch.uint8, device=device)
            self._scratch_buf = buf
        return buf

    def _fused_gate_up(self):
        """把 w1/w3 拼成一个 [2N, K] 的 fp8 权重（vLLM 的 gate_up_proj）。

        shared expert 每层都要跑，而它在 decode 里是三次独立 GEMV 加一次 cat：
        w1(x) 和 w3(x) 读同一个 x、同一个量化后的激活，却各自 launch 一次、
        各自量化一遍，结果还要 torch.cat 拼起来交给 silu_and_mul。
        拼成一个权重之后是一次 GEMV 直接产出 [2N]，silu_and_mul 直接吃，
        连 cat 都省了。
        """
        if getattr(self, "_gu_w", None) is not None:
            return self._gu_w, self._gu_s
        a, b = self.w1, self.w3
        if not (hasattr(a, "weight") and hasattr(b, "weight")):
            return None, None
        if (a.weight.dtype != b.weight.dtype or a.scale.dtype != b.scale.dtype
                or int(a.weight.shape[1]) != int(b.weight.shape[1])
                or int(a.weight.shape[0]) % 32 or int(b.weight.shape[0]) % 32):
            return None, None
        self._gu_w = torch.cat([a.weight, b.weight], dim=0).contiguous()
        self._gu_s = torch.cat([a.scale, b.scale], dim=0).contiguous()
        return self._gu_w, self._gu_s

    def forward(self, x: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
        from .vllm_ops import clamp_swiglu
        _gw, _gs = self._fused_gate_up()
        if _gw is not None and x.is_cuda and x.dtype == torch.bfloat16:
            from ..gemm import fp8_linear
            _gu = fp8_linear(x, _gw, _gs)
            y = clamp_swiglu(_gu, self.swiglu_limit)
        elif hasattr(self.w1, "forward"):  # fp8 QuantLinear shared expert
            gate = self.w1.forward(x)
            up = self.w3.forward(x)
            y = clamp_swiglu(torch.cat((gate, up), dim=-1), self.swiglu_limit)
        else:
            x = x.to(self.w1.dtype)
            gate = F.linear(x, self.w1)
            up = F.linear(x, self.w3)
            y = clamp_swiglu(torch.cat((gate, up), dim=-1), self.swiglu_limit)
        if y is None:
            raise RuntimeError("vllm clamp_swiglu unavailable (vllm_silu_and_mul.cubin)")
        if weights is not None:
            y = weights * y
        if hasattr(self.w2, "forward"):
            return self.w2.forward(y)
        return F.linear(y.to(self.w2.dtype), self.w2)


class MoE:
    def _scratch(self, nbytes: int, device):
        """MoE 融合 kernel 的中间 buffer（fp8 激活 + e8m0 scale），按大小复用。"""
        buf = getattr(self, "_scratch_buf", None)
        if buf is None or buf.numel() < nbytes or buf.device != device:
            buf = torch.empty(nbytes, dtype=torch.uint8, device=device)
            self._scratch_buf = buf
        return buf

    def _ts_offsets(self, layer: int):
        """(w1,s1,w3,s3,w2,s2) 在 blob 内的字节偏移；同一层所有专家同布局，按层缓存。"""
        cache = getattr(self, "_ts_off_cache", None)
        if cache is None:
            cache = self._ts_off_cache = {}
        hit = cache.get(layer)
        if hit is not None:
            return hit
        idx = getattr(getattr(self, "_expert_store", None), "index", None)
        if idx is None:
            raise RuntimeError("MoE: expert_store.index 不可用")
        t = idx.expert_blob(layer, 0).tensors
        hit = cache[layer] = (
            t["w1.weight"].dest_offset, t["w1.scale"].dest_offset,
            t["w3.weight"].dest_offset, t["w3.scale"].dest_offset,
            t["w2.weight"].dest_offset, t["w2.scale"].dest_offset)
        return hit

    def _forward_gpu_only(self, x: torch.Tensor, xf: torch.Tensor,
                          indices: torch.Tensor, weights: torch.Tensor):
        """整层不碰 CPU 的 decode 路径；走不通返回 None，由调用方回落慢路径。

        自包含：只吃 x / xf / indices / weights 和槽位表，不引用 forward 后面才定义的
        任何变量（上一版「挪一处崩一处」正是因为这个）。量化、累加 buffer 都在这里建。
        """
        if not (xf.is_cuda and xf.size(0) == 1):
            self._gpu_fail = "line 364"
            return None
        sl = getattr(getattr(self, "_expert_store", None), "slots", None)
        if sl is not None and not getattr(sl, "_arena", None):
            try:
                sl._ensure()
            except Exception:
                pass
        tab = getattr(sl, "_slot_tab", None) if sl is not None else None
        arena = getattr(sl, "_arena", None) if sl is not None else None
        if not arena or len(arena) != 1:
            self._gpu_fail = "line 375"
            return None
        if tab is None:
            # 表是懒建的。warmup 阶段还没有路由统计，先建全 -1，miss 分支会填。
            tab = sl.slot_table(int(getattr(self, "_n_layers_hint", 40) or 40), 384)
        try:
            from .fp4_native import quantize_act
            from ..kernels import native

            aq, asc = quantize_act(xf, xf.size(-1))
            # 路由计数（pin 用）。原来在 serve() 里更新，那条路删掉后这里补上：
            # 只在还没 pin 的时候统计，pin 之后路由恒定，不必再花这次 D2H。
            if (not getattr(self, "_all_pinned", False)
                    and not torch.cuda.is_current_stream_capturing()):
                _rc = self._route_counts
                for _e in indices.view(-1).tolist():
                    _rc[int(_e)] = _rc.get(int(_e), 0) + 1
            sids = tab[self.layer_id].index_select(0, indices.view(-1))
            # miss 自己补：缺的专家当场 fetch 进槽位，重建 sids。同一条路，不是第二套。
            if not getattr(self, "_all_pinned", False) and bool((sids < 0).any().item()):
                _miss = sorted({int(v) for v, sid in
                                zip(indices.view(-1).tolist(), sids.tolist()) if int(sid) < 0})
                if _miss:
                    self._expert_store.fetch(self.layer_id, _miss)
                    sl.table_sync()
                    tab = sl._slot_tab
                    sids = tab[self.layer_id].index_select(0, indices.view(-1))
                if bool((sids < 0).any().item()):
                    raise RuntimeError(
                        f"MoE layer {self.layer_id}: {_miss} still not resident")
            n_exp = int(indices.numel())
            nmid = int(self.cfg.moe_inter_dim)
            blob = int(sl.blob_bytes)
            off = self._ts_offsets(self.layer_id)
            scr = self._scratch(n_exp * (nmid + (nmid >> 5)), arena[0].device)
            y = getattr(self, "_gpu_y_buf", None)
            if y is None or y.shape != xf.shape or y.device != xf.device:
                y = torch.zeros(xf.shape, dtype=torch.float32, device=xf.device)
                self._gpu_y_buf = y
            else:
                y.zero_()

            fa = _MOE_FUSED.get("act")
            if fa is None:
                fa = native.BoundLaunch(native.kernel("moe_fused", "moe_fused_act"),
                                        ((nmid + 31) // 32, 1), (512,), "TTTTTOOiiiiiiiif")
                _MOE_FUSED["act"] = fa
            fa(aq.reshape(-1), asc.reshape(-1), arena[0], sids, weights[0],
               scr[:n_exp * nmid], scr[n_exp * nmid:],
               n_exp, nmid, int(self.cfg.dim), blob,
               off[0], off[1], off[2], off[3], float(self.swiglu_limit), stream=None)

            fd = _MOE_FUSED.get("down")
            if fd is None:
                fd = native.BoundLaunch(native.kernel("moe_fused", "moe_fused_down"),
                                        ((int(self.cfg.dim) + 15) // 16,), (512,),
                                        "TTTTOiiiiiii")
                _MOE_FUSED["down"] = fd
            fd(scr[:n_exp * nmid].view(torch.float8_e4m3fn), scr[n_exp * nmid:],
               arena[0], sids, y,
               n_exp, int(self.cfg.dim), nmid, blob, off[4], off[5], 0, stream=None)
            return y.view(*x.shape[:-1], x.shape[-1]).to(x.dtype)
        except Exception as _exc:
            if __import__("os").environ.get("TI_GPUDBG") == "1":
                import traceback as _tb
                print(f"[gpufail] {_exc!r}", flush=True)
                _tb.print_exc()
            return None

    def __init__(self, store: ShardStore, layer_id: int, cfg: TinyConfig, expert_store=None,
                 max_bf16_experts: int = 8, topk: int = 0, cache_shared_bf16: bool = False,
                 plan_fn=None):
        self.cfg = cfg
        self.layer_id = layer_id
        self._expert_store = expert_store  # optional tinyinfer.loading.ExpertStore
        # 0 = never keep bf16 copies (production/GPU path); N>0 = bounded reference cache
        self._max_bf16 = int(max_bf16_experts)
        self.gate = Gate(store, layer_id, cfg, topk=topk)
        p = f"layers.{layer_id}.ffn."
        # shared expert stays fp8 on the device (halves its VRAM; official fp8 GEMM path)
        from ..linear import QuantLinear

        def _mk(name, cache):
            """One shared-expert projection, split across cards when a plan says so."""
            ql = QuantLinear(store.get(p + name + ".weight"), store.get(p + name + ".scale"),
                             cache_bf16=cache)
            if plan_fn is None:
                return ql
            plan = plan_fn("shared." + name, int(ql.weight.shape[0]))
            if not plan or len(plan) <= 1:
                return ql
            from ..tp import shard_quant_linear
            return shard_quant_linear(ql.weight, ql.scale, plan)

        # The shared expert runs on EVERY token and on every layer, unlike a routed expert:
        # 3 projections x 40 layers = 120 launches per decode step that always happen.
        # Splitting its output dimension across cards divides those reads by the number of
        # cards with no reduction (the slices are disjoint).
        self.shared = Expert(
            _mk("shared_experts.w1", cache_shared_bf16),
            _mk("shared_experts.w2", cache_shared_bf16),
            _mk("shared_experts.w3", cache_shared_bf16),
            cfg.swiglu_limit,
        )
        self._store = store
        self._expert_cache: dict[int, Expert] = {}
        self._last_ids: list[int] | None = None
        # 路由频率统计：用来决定哪些专家值得固定进显存槽位。decode 每 token 过
        # 240 个 expert，1359 个槽位装得下热点 —— 一旦 pin 住，路由就恒命中，
        # 不必每层再 tolist() 回 CPU 决定搬哪些（实测 3.3ms/层、130ms/token）。
        self._route_counts: dict[int, int] = {}
        # 静态 decode（CUDA graph 捕获用）：为 True 时不回 CPU 取路由，
        # 用 _static_active 里预先量好的专家集合。
        self._static_decode: bool = False
        self._static_active: list[int] = []
        self.swiglu_limit = cfg.swiglu_limit
        self._fused_ok: bool | None = None

    def _expert_from(self, get) -> Expert:
        """Build an Expert from a name->tensor accessor, packed fp4 when on CUDA."""
        target = self.gate.weight.device
        if target.type == "cuda":
            e = Expert(None, None, None, self.swiglu_limit)
            # non_blocking: the source is a slice of the pinned host pool, so a blocking
            # copy has nothing to gain and everything to lose -- it waits for the whole
            # transfer queue to drain before returning, once per tensor. This path runs
            # 6 copies per expert and a decode token touches 660 experts, so on a
            # slots=0 plan that is ~3960 synchronous waits per token; the docstring on
            # _expert_from_rec measured ~18ms/layer from exactly this, i.e. ~0.7s of a
            # 1.6s step. Pinned->device with non_blocking is a plain async DMA and the
            # caller's stream ordering still makes the bytes visible before use.
            w1t = get("w1.weight").to(target, non_blocking=True)
            s1t = get("w1.scale").to(target, non_blocking=True)
            w2t = get("w2.weight").to(target, non_blocking=True)
            s2t = get("w2.scale").to(target, non_blocking=True)
            w3t = get("w3.weight").to(target, non_blocking=True)
            s3t = get("w3.scale").to(target, non_blocking=True)
            e.fp4 = (w1t, s1t, w2t, s2t, w3t, s3t)
            return e
        return Expert(
            dequant_fp4_block(get("w1.weight"), get("w1.scale")),
            dequant_fp4_block(get("w2.weight"), get("w2.scale")),
            dequant_fp4_block(get("w3.weight"), get("w3.scale")),
            self.swiglu_limit,
        )

    def _to_acc(self, val, y):
        """Bring an expert's output back to the device the accumulator lives on.

        With the slot arenas spread over several cards an expert can be computed on a
        different device than the accumulator; the sum has to happen on one of them. On a
        single card both are already the same device and this returns the tensor
        unchanged, so there is no copy and no branch on the card count.
        """
        if val is None or not torch.is_tensor(val):
            return val
        if val.device != y.device:
            return val.to(y.device, non_blocking=True)
        return val

    def _expert_from_rec(self, rec) -> Expert:
        """One cudaMemcpyAsync per expert, straight from the host-pool slice.

        Six per-tensor copy_() calls per expert is 36 synchronous pageable copies per
        layer: the link is PCIe 5.0 x8 (measured 26GB/s), but each blocking copy waits for
        the queue to drain, which put ~18ms/layer on the critical path. One async 18.8MB
        DMA is 0.7ms and never blocks.
        """
        # The device this expert should live on. `gate.weight.device` (one fixed card) was
        # the only possibility when every expert fit on that card; with the slot arenas
        # spread over the pool, an expert belongs wherever ITS slot was allocated, so ask
        # the record rather than assuming. A single card answers the same device either
        # way, which is why this replaces the constant instead of special-casing it.
        target = self.gate.weight.device
        # The slot buffer IS the storage: its device is where this expert's bytes already
        # are, so reading rec.buffer.device answers the question without trusting a
        # separately-maintained string (which lags a re-allocation). A host buffer means
        # the RAM-cache path, where the gate's device is the right destination.
        _buf = getattr(rec, "buffer", None)
        if _buf is not None and getattr(_buf, "device", None) is not None                 and _buf.device.type == "cuda":
            target = _buf.device
        if target.type == "cuda" and hasattr(rec, "tensors_on"):
            t = rec.tensors_on(target)
            e = Expert(None, None, None, self.swiglu_limit)
            e.fp4 = (t["w1.weight"], t["w1.scale"], t["w2.weight"], t["w2.scale"],
                     t["w3.weight"], t["w3.scale"])
            # ExpertRecord.key is (layer, expert): exactly the identity the marlin repack
            # cache needs, and stable across the blob ring reusing its address.
            e.key = getattr(rec, "key", None)
            return e
        return self._expert_from(rec.tensor)

    def _make_expert(self, rec) -> Expert:
        """Materialise one expert for the gate's device: fp4 stays packed on CUDA.

        Dequantising on the host costs 74ms per expert (measured), which would dominate
        any CPU-side path, so the packed fp4 tensors go to the device and the fp4 kernel
        consumes them directly.
        """
        return self._expert_from_rec(rec)

    def _expert(self, i: int) -> Expert:
        e = self._expert_cache.get(i)
        if e is None:
            if self._expert_store is not None:
                with self._expert_store.experts(self.layer_id, [i]) as recs:
                    e = self._make_expert(recs[0])
            else:
                p = f"layers.{self.layer_id}.ffn.experts.{i}."
                e = self._expert_from(lambda n: self._store.get(p + n))
            self._expert_cache[i] = e
        return e

    def to(self, device):
        self.shared.to(device)
        self.gate.weight = self.gate.weight.to(device)
        self.gate.bias = self.gate.bias.to(device)
        if self.gate.bias_vl is not None:
            self.gate.bias_vl = self.gate.bias_vl.to(device)
        return self

    def prefetch(self, expert_ids) -> None:
        """Warm RAM and stage predicted experts on VRAM (learned predictor: ~95% hit)."""
        if self._expert_store is not None and expert_ids:
            try:
                self._expert_store.prefetch(self.layer_id, expert_ids)
            except Exception:
                pass

    def forward(self, x: torch.Tensor, image_mask: torch.Tensor | None = None) -> torch.Tensor:
        import os as _os0
        _prof = _os0.environ.get("TI_MOE_PROF") == "1"
        import time as _tp
        _t_start = _tp.perf_counter() if _prof else 0.0
        self._parts = getattr(self, "_parts", None)
        if _prof and self._parts is None:
            self._parts = {"gate": 0.0, "bincount": 0.0, "sel": 0.0, "serve": 0.0,
                           "fetch": 0.0, "shared": 0.0, "alloc": 0.0}
        shape = x.shape
        xf = x.view(-1, self.cfg.dim)
        _t1 = _tp.perf_counter() if _prof else 0.0
        weights, indices = self.gate.forward(xf, None if image_mask is None else image_mask.flatten())
        # 整层不碰 CPU 的 decode 路径：槽位表齐了就接管，否则返回 None 回落慢路径。
        _gpu_only = self._forward_gpu_only(x, xf, indices, weights)
        if _gpu_only is not None:
            return _gpu_only
        # eager / serve 路径已整段删除：MoE 只有纯 GPU 这一条路。
        raise RuntimeError(
            f"MoE layer {self.layer_id}: pure-GPU path unavailable "
            f"[{getattr(self, '_gpu_fail', '?')}] "
            f"(cuda={xf.is_cuda} rows={xf.size(0)})")
