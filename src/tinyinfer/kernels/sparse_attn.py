"""Sparse attention dispatch -- CUDA only.

The TileLang implementation this module used to wrap is gone. Everything goes
through the sparse_attn_sink cubin (kernels/cuda/sparse_attn_sink.cu), with the
PyTorch reference kept solely as the baseline the tests compare against.

No silent fallback: when the cubin is unavailable this raises.
"""
import torch

from ..kernels import native


def sparse_attn_cuda(q, kv, sink, idxs, scale):
    """out[q,h,:] = sum_i exp(s_i) v[slot_i] / (sum_i exp(s_i) + exp(sink_h)),

    q    [B, M, H, D] bf16
    kv   [B, N, D]    bf16
    sink [H]          fp32
    idxs [B, M, K]    int32 (-1 = empty slot)
    """
    b, m, h, d = q.shape
    n, k = kv.size(1), idxs.size(-1)
    out = torch.empty_like(q)
    # kernel 用 extern __shared__：s_m / s_l 各 nwarps 个 float，s_out 是
    # nwarps * kMaxLocal(=16) 个 float。不给 shared 大小，sm 就指向非法地址，
    # cuLaunchKernel 直接报 700（illegal address）。
    block = 128
    nwarps = block // 32
    shared = (2 * nwarps + nwarps * 16) * 4
    native.launch(
        native.kernel("sparse_attn_sink", "sparse_attn_sink"),
        (b * m * h,), (block,),
        [q, kv, sink, idxs, out, b, m, h, d, n, k, float(scale)],
        shared=shared,
    )
    return out


def sparse_attn_dispatch(q, kv, sink, idxs, scale):
    """Single entry point: CUDA kernel on GPU, reference on CPU."""
    if q.is_cuda and q.dtype == torch.bfloat16:
        return sparse_attn_cuda(q, kv, sink, idxs, scale)
    from ..nn import sparse_attn_ref
    return sparse_attn_ref(q, kv, sink, idxs, scale)
