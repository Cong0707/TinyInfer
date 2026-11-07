"""Minimal numeric building blocks used by the reference forward."""
from __future__ import annotations

import math
import os

import torch
import torch.nn.functional as F


_RMS_NATIVE = {"ok": None}


def _rms_native_ok() -> bool:
    if _RMS_NATIVE["ok"] is None:
        try:
            from .kernels import native
            native.kernel("rms_norm", "rms_norm_kernel")
            _RMS_NATIVE["ok"] = True
        except Exception as _exc:
            # 不回退。torch 版本是 7 个 launch，每层 2 次、40 层 = 每 token 560 个
            # 额外 launch，而失败本身会被这个 latch 永久记住且只在环境变量下可见。
            # 探测失败就该直接报错。
            raise RuntimeError(
                f"rms_norm native kernel unavailable: {_exc!r}") from _exc
    return _RMS_NATIVE["ok"]


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """HF-style RMSNorm: normalize in fp32, scale by the stored weight.

    One kernel for the whole chain. The torch form is float/square/mean/rsqrt/mul/mul/to
    -- seven launches, twice per layer, 560 per decode step for a 1-row activation. The
    kernel keeps the same fp32 reduction and the same final bf16 cast.
    """
    orig_dtype = x.dtype
    # vLLM's rms_norm runs at 321 GB/s here; the native kernel below launches one
    # 32-thread block per row and measured 1.44x off at M=1 and 3.53x at M=8.
    try:
        from .deepseek.vllm_ops import rms_norm as _vllm_rms_norm

        _hit = _vllm_rms_norm(x, weight, eps)
        if _hit is not None:
            return _hit.to(orig_dtype) if _hit.dtype != orig_dtype else _hit
    except Exception:
        pass
    if (x.is_cuda and x.dtype == torch.bfloat16 and _rms_native_ok()
            and weight.numel() == x.shape[-1]):
        try:
            from .kernels import native
            shape = x.shape
            d = shape[-1]
            rows = 1
            for v in shape[:-1]:
                rows *= int(v)
            xc = x.reshape(rows, d).contiguous()
            wf = weight.float().contiguous()
            y = torch.empty((rows, d), dtype=torch.bfloat16, device=x.device)
            native.launch(native.kernel("rms_norm", "rms_norm_kernel"), (rows,), (32,),
                          [xc, wf, y, rows, d, float(eps)])
            return y.view(*shape)
        except Exception as _exc:
            # 不回退到下面那条 7 个算子的 torch 链：那条每层 2 次、40 层就是
            # 每 token 560 个额外 launch，而且只在速度上体现，没有任何提示。
            raise RuntimeError(f"rms_norm native launch failed: {_exc!r}") from _exc
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (y * weight.float()).to(orig_dtype)


def window_idxs_prefill(seqlen: int, window: int, device=None) -> torch.Tensor:
    """Causal sliding-window slot indexes for a fresh prefill (see official get_window_topk_idxs)."""
    end = torch.arange(seqlen, device=device).unsqueeze(1)
    idxs = (end - window + 1).clamp(0) + torch.arange(min(seqlen, window), device=device)
    idxs = torch.where(idxs > end, torch.full_like(idxs, -1), idxs)
    return idxs.int()


def window_idxs_general(start_pos: int, seqlen: int, old_len: int, window: int, device=None) -> torch.Tensor:
    """Window slots for a chunk inside a linear KV history.

    kv_att = [old_history (old_len, oldest first), new_chunk (seqlen)]; a query at global
    position p attends to [max(0, p-window+1) .. p]. Returns [seqlen, window] int32 (-1 = empty).
    """
    qpos = torch.arange(start_pos, start_pos + seqlen, device=device)
    back = torch.arange(window - 1, -1, -1, device=device)
    gpos = qpos[:, None] - back[None, :]  # [s, window] window start..p
    base = start_pos - old_len
    idx = gpos - base
    valid = (gpos >= 0) & (idx >= 0) & (idx < old_len + seqlen)
    return torch.where(valid, idx, torch.full_like(idx, -1)).int()


def window_idxs_ring(start_pos: int, seqlen: int, old_len: int, window: int, device=None) -> torch.Tensor:
    """环形 cache 的窗口槽位：位置 p 存在 p % window 这一格。

    vLLM 的 paged cache 也是这个形状 —— 缓存一旦分配就是 fixed 的 slot 池，写入
    按 slot_mapping 落到某一格，读取按 block table 取，历史永远不搬。这里的
    slot 就是 position % window，所以窗口索引只依赖绝对位置，不再依赖「历史
    被移动过多少」（旧实现每 token 都要 buf[:, :n_old].copy_(old.clone()) 把整个
    历史左移一格）。

    Returns [seqlen, window] int32, -1 = 还没写过的位置。
    """
    back = torch.arange(window - 1, -1, -1, device=device)
    # start_pos 给 tensor 也能算：arange 只生成 [0, seqlen)，位置靠广播加进去。
    base = start_pos if torch.is_tensor(start_pos) else torch.tensor(start_pos, device=device)
    gpos = (base + torch.arange(seqlen, device=device))[:, None] - back[None, :]
    idx = gpos % window
    valid = gpos >= 0
    return torch.where(valid, idx, torch.full_like(idx, -1)).int()


# 窗口索引按 (起始位置, s, window) 缓存：一个 decode step 里 40 层看到的是同一组
# 槽位，而生成它要 arange/remainder/ge/full_like/where 五个 kernel。旧代码只有
# 「窗口刚好满」这一种情况才复用 _decode_idxs，窗口没满的那几百个 token 里每层
# 都重算一遍 —— 40 层 x 5 个 kernel，纯粹是重复劳动。
_RING_IDX: dict = {}


def window_idxs_ring_cached(start_pos: int, seqlen: int, window: int, device) -> torch.Tensor:
    key = (int(start_pos), int(seqlen), int(window), str(device))
    hit = _RING_IDX.get(key)
    if hit is not None:
        return hit
    if len(_RING_IDX) > 8192:
        _RING_IDX.clear()
    t = window_idxs_ring(start_pos, seqlen, 0, window, device)
    _RING_IDX[key] = t
    return t


def window_idxs_per_row(positions, seqlen: int, old_len, window: int, device=None) -> torch.Tensor:
    """Window slots for a batch whose rows are at different depths.

    window_idxs_general() describes ONE row and is then broadcast over the batch. Two
    things then go wrong for rows at different depths, and together they are the whole
    reason a batched call disagreed with the same sequences run one at a time:

      * the absolute position differs, so each row must derive its own query positions;
      * the history length differs too -- a row that just arrived has old_len 0 while one
        already decoding has old_len == window. One shared old_len indexes the wrong slot
        for every row but one (old_len=4 gives row 1 [1,2,3,4] while row 0 wants
        [-1,-1,-1,0]).

    old_len may be an int or a per-row sequence; this builds [b, seqlen, window].
    positions: [b] absolute position of each row's first column in this call.
    """
    back = torch.arange(window - 1, -1, -1, device=device)
    old_lens = (list(old_len) if not isinstance(old_len, int) else [old_len] * len(positions))
    out = []
    for p0, ol in zip(positions.tolist(), old_lens):
        ol = int(ol)
        qpos = torch.arange(int(p0), int(p0) + seqlen, device=device)
        gpos = qpos[:, None] - back[None, :]
        idx = gpos - (int(p0) - ol)
        valid = (gpos >= 0) & (idx >= 0) & (idx < ol + seqlen)
        out.append(torch.where(valid, idx, torch.full_like(idx, -1)).int())
    return torch.stack(out, 0)


def sparse_attn_ref(
    q: torch.Tensor, kv: torch.Tensor, sink: torch.Tensor, idxs: torch.Tensor, scale: float
) -> torch.Tensor:
    """Naive gather-sparse MQA attention with a learned attn-sink logit per head.

    q: [b, s, h, d]; kv: [b, n, d]; sink: [h]; idxs: [b, s, t] (-1 = empty slot).
    Equivalent to the tilelang sparse_attn kernel: scores = q.k / sqrt(d), softmax over
    gathered slots PLUS one virtual sink slot whose logit is attn_sink[h].
    """
    b, s, h, d = q.shape
    t = idxs.shape[-1]
    valid = idxs >= 0  # [b, s, t]
    safe = idxs.clamp(min=0)  # [b, s, t]
    # gather kv for every query: [b, s, t, d]
    flat = safe.reshape(b, s * t, 1).expand(b, s * t, d)
    k = kv.gather(1, flat).reshape(b, s, t, d)
    scores = torch.einsum("bshd,bstd->bsht", q.float(), k.float()) * scale  # [b, s, h, t]
    scores = scores.masked_fill(~valid.unsqueeze(2), float("-inf"))
    max_scores = scores.max(dim=-1).values  # [b,s,h]
    # fold the sink in as an extra logit; a finite bound keeps fully-empty rows finite (all-zero out)
    max_final = torch.max(max_scores, sink.view(1, 1, h))
    p = torch.exp(scores - max_final.unsqueeze(-1))
    denom = p.sum(-1) + torch.exp(sink.view(1, 1, h) - max_final)
    p = p / denom.unsqueeze(-1)
    o = torch.einsum("bsht,bstd->bshd", p, k.float())
    return o.to(q.dtype)


def softmax_sample(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Greedy multinomial sampling (same math as the official sample())."""
    if temperature == 0:
        return logits.argmax(dim=-1)
    logits = logits / max(temperature, 1e-5)
    probs = torch.softmax(logits.float(), dim=-1)
    return probs.div_(torch.empty_like(probs).exponential_(1)).argmax(dim=-1)


def sample_logits(logits: torch.Tensor, temperature: float = 0.0, top_p: float = 1.0,
                  top_k: int = 0, repetition_penalty: float = 1.0,
                  recent_tokens: list[int] | None = None) -> int:
    """Greedy (temperature 0) or temperature/top-p/top-k sampling.

    repetition_penalty / recent_tokens follow the vLLM sampler convention: a token
    already present in the recent window gets its (positive/negative) logit divided/
    multiplied by the penalty. The default 1.0 disables it.
    """
    if repetition_penalty != 1.0 and recent_tokens:
        idx = torch.tensor(sorted(set(recent_tokens)), dtype=torch.long, device=logits.device)
        vals = logits.index_select(-1, idx).float()
        vals = torch.where(vals > 0, vals / repetition_penalty, vals * repetition_penalty)
        logits = logits.clone()
        logits.index_copy_(-1, idx, vals.to(logits.dtype))
    if temperature <= 0:
        return int(logits.argmax(dim=-1))
    logits = logits.float() / temperature
    if top_k and top_k > 0:
        k = min(top_k, logits.numel())
        v, _ = torch.topk(logits, k)
        logits = logits.masked_fill(logits < v[-1], float("-inf"))
    probs = torch.softmax(logits, dim=-1)
    if top_p < 1.0:
        sorted_p, order = torch.sort(probs, descending=True)
        cum = torch.cumsum(sorted_p, dim=-1)
        keep = cum - sorted_p < top_p
        sorted_p = torch.where(keep, sorted_p, torch.zeros_like(sorted_p))
        probs = torch.zeros_like(probs).scatter_(0, order, sorted_p)
        probs = probs / probs.sum().clamp_min(1e-12)
    return int(torch.multinomial(probs, 1))


_ROPE_FN = None
_ROTARY_CACHE: dict = {}
_POS_CACHE: dict = {}


def _rope_kernel():
    """vLLM pos_encoding_kernels.cu 的 rotary_embedding（移植版）。"""
    global _ROPE_FN
    if _ROPE_FN is None:
        from .kernels import native
        _ROPE_FN = native.kernel("vllm_rotary", "vllm_rotary_embedding_bf")
    return _ROPE_FN


def _rotary_cache(freqs_cis: torch.Tensor, npos: int, rot: int) -> torch.Tensor:
    """把 complex 的 freqs_cis 换成 vLLM 的 cos_sin_cache 布局 [npos, rot]。

    上游的约定是「每个 position 一段 rot 长的 float，前一半 cos、后一半 sin」，
    kernel 里 cos_ptr = cache_ptr、sin_ptr = cache_ptr + rot/2。
    这张表在一次 decode 里不变，按 (存储地址, npos, rot) 缓存。
    """
    key = (freqs_cis.data_ptr(), int(npos), int(rot))
    hit = _ROTARY_CACHE.get(key)
    if hit is not None:
        return hit
    if len(_ROTARY_CACHE) > 256:
        _ROTARY_CACHE.clear()
    fr = torch.view_as_real(freqs_cis.reshape(npos, rot // 2))
    hit = torch.cat([fr[..., 0], fr[..., 1]], dim=-1).float().contiguous()
    _ROTARY_CACHE[key] = hit
    return hit


def _pos_arange(npos: int, device) -> torch.Tensor:
    key = (int(npos), str(device))
    hit = _POS_CACHE.get(key)
    if hit is None:
        hit = _POS_CACHE[key] = torch.arange(npos, device=device, dtype=torch.int64)
    return hit


def _apply_rope_cuda(x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool):
    """原地旋转。返回 None 表示这个形状走不了（调用方用 torch 版）。

    调用方传的是 q[..., -rope_dim:] 这种切片，所以 rope 段在 x 的偏移是 0，
    而 head 之间隔着 head_dim（x 的 stride(-2)）。kernel 直接按这两个 stride
    读写，不需要把切片拷成连续张量。
    """
    if not x.is_cuda or x.dtype != torch.bfloat16 or x.dim() < 2:
        return None
    rot = int(x.shape[-1])
    if rot <= 0 or (rot & 1) or x.stride(-1) != 1:
        return None
    if freqs_cis.dim() < 2:
        return None
    npos = int(freqs_cis.shape[0])
    if npos <= 0:
        return None
    heads = int(x.shape[-2])
    hstride = int(x.stride(-2))
    ntokens = 1
    for v in x.shape[:-2]:
        ntokens *= int(v)
    if ntokens != npos:
        return None
    qstride = int(x.stride(-3)) if x.dim() >= 3 else heads * hstride
    if qstride < hstride or hstride < rot:
        return None
    from .kernels import native
    native.launch(_rope_kernel(), (npos,), (256,),
                  [_pos_arange(npos, x.device), x, _rotary_cache(freqs_cis, npos, rot),
                   rot, qstride, hstride, heads, rot, 0, 1 if inverse else 0])
    return x


def apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """Rotate adjacent element pairs as complex numbers; accepts [b,s,d] or [b,s,h,d].

    Used both for queries/keys and, with inverse=True, to un-rotate the attention output.
    """
    _r = _apply_rope_cuda(x, freqs_cis, inverse)
    if _r is not None:
        return _r
    y = x
    xc = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()
    if xc.ndim == 3:
        f = freqs_cis.view(1, xc.size(1), xc.size(-1))
    else:
        f = freqs_cis.view(1, xc.size(1), 1, xc.size(-1))
    out = torch.view_as_real(xc * f).flatten(-2)
    y.copy_(out)
    return y


def precompute_freqs_cis(
    dim: int, seqlen: int, original_seq_len: int, base: float, factor: float, beta_fast: float, beta_slow: float
) -> torch.Tensor:
    """Rotary frequencies (complex exponentials), one row per position; YaRN when original_seq_len > 0."""
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    mscale = 1.0
    if original_seq_len > 0:
        def corrected_dim(rotations: float) -> float:
            return dim * math.log(original_seq_len / (rotations * 2 * math.pi)) / (2 * math.log(base))

        low = max(math.floor(corrected_dim(beta_fast)), 0)
        high = min(math.ceil(corrected_dim(beta_slow)), dim - 1)
        ramp = ((torch.arange(dim // 2, dtype=torch.float32) - low) / max(high - low, 1e-3)).clamp(0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
        # DeepSeek's own serving stack (DeepInfer rotary_embedding.py) applies
        # yarn_get_mscale(factor) = 0.1 * mscale * ln(factor) + 1 to the cos/sin cache,
        # with mscale defaulting to True and mscale_all_dim to 0. The model author is the
        # authority here, so the magnitude scaling is part of the reference semantics.
        # DeepInfer (the model author's serving stack) applies yarn_get_mscale:
        # 0.1*ln(factor)+1 = 1.277 at factor 16, and the author's stack is the authority
        # this engine aligns to. The plain HF reference table omits it, which is why
        # docs/parity.md records the DeepInfer comparison as the reference one.
        _ms = float(os.environ.get("TI_YARN_MSCALE", "1.0"))
        if factor > 1 and _ms:
            mscale = 0.1 * _ms * math.log(factor) + 1.0
    freqs = torch.outer(torch.arange(seqlen), freqs)
    return torch.polar(torch.ones_like(freqs), freqs) * mscale