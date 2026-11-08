"""vLLM 风格的 w4a8 GEMV 封装（fp8 激活 x fp4 权重）。

对应 vLLM 的 `cutlass_scaled_mm` / `moe_wna16_gemm` 那类 C++ 入口：
Python 这一侧只负责把张量摆成 kernel 需要的布局，算术全在 cubin 里。
kernel 本体在 kernels/cuda/gemv_fp4_w4a8.cu（照 vLLM 的 machete/w4a8 布局写的）。

没有 TileLang 可选项，失败就抛异常。
"""
from __future__ import annotations

import torch

from . import native

_BOUND: dict = {}


def gemv_fp4_w4a8(aq: torch.Tensor, asc: torch.Tensor,
                  w_fp4: torch.Tensor, w_scale: torch.Tensor,
                  _ws_unused=None) -> torch.Tensor:
    """Y[M,N] = A[M,K](fp8) * W[N,K](fp4, e8m0 per-32 scales)^T.

    aq      [M, K]     float8_e4m3fn
    asc     [M, K/32]  float（e8m0 转成 fp32）
    w_fp4   [N, K/2]   uint8 打包的 E2M1
    w_scale [N, K/32]  float
    """
    if aq.dim() == 1:  # MoE 路径把单行激活摊平成 [K]
        aq = aq.unsqueeze(0); asc = asc.unsqueeze(0)
    m, k = aq.shape
    n = w_fp4.shape[0]
    y = torch.empty((m, n), dtype=torch.bfloat16, device=aq.device)
    key = (int(m), int(n), int(k))
    fn = _BOUND.get(key)
    if fn is None:
        # ROWS=4：每个 block 算 4 行输出，grid = M * ceil(N/4)
        rows = 4
        nb = (n + rows - 1) // rows
        fn = native.BoundLaunch(
            native.kernel("gemv_fp4_w4a8", "gemv_fp4_w4a8"),
            (m * nb,), (128,), "TTTTOiii",
        )
        _BOUND[key] = fn
    fn(aq, asc, w_fp4, w_scale, y, int(m), int(n), int(k))
    return y
