"""Decode-time fp4 MoE fused across every routed expert of a layer.

The tilelang path launches one GEMV per expert per projection, converts each expert's
e8m0 scale block to fp32 before the kernel can read it, and then spends about seven
more launches on clamp/silu/scale. At decode that is ~34 launches per layer; the scale
copies alone measured 205ms of a 790ms decode step.

'native/moe_gemv6.cu' decodes the fp4 weights, the e8m0 scale bytes and the fp8 e4m3
activation inline, with the expert index carried by grid.y, so one layer is four
launches and no conversion pass at all:

    gu = gate_up(aq, asc, w1, s1, w3, s3)   # [E, 2N]  fp8 e4m3 activation
    a  = act(gu, w_exp, limit)              # [E,  N]  clamp+silu+routing weight
    d  = down(a, w2, s2)                    # [E, 2D]  bf16 activation, no quantise
    acc(d, y, ...)                          # y += sum_e d[e]

Measured on synthetic weights at the real shapes (6 experts, w1/w3 2304x5120):
fused gate_up 0.225-0.279ms against 1.011ms for the twelve tilelang launches it
replaces, and the result matches an fp32 reference to within one bf16 ulp.
"""
import torch

from ..kernels import native

MAX_EXPERTS = 8
_fn_cache: dict[str, object] = {}
_ones: dict[int, torch.Tensor] = {}


def available() -> bool:
    """True when the hand-written kernels are built.

    不回退成 False。这个函数的返回值被 moe.py 的 _fused_ok 永久 latch，
    一次失败就等于本进程再也走不到 fused MoE：每层从 4 个 launch 变成 ~34 个，
    多行时还会换成另一套精度不同的算术（实测同 prompt 相差 16.85）。
    而且这个 latch 只在 TI_FUSED_STATS=1 时才有一行输出。
    """
    _kernel("moe_gemv6")
    return True


def _kernel(name: str):
    k = _fn_cache.get(name)
    if k is None:
        k = _fn_cache[name] = native.kernel("moe_gemv6", name)
    return k


def supported_shape(n_rows: int, k_dim: int) -> bool:
    """The GEMV walks k_dim in 256-wide steps and needs a whole 16-row tile."""
    return k_dim > 0 and k_dim % 256 == 0 and n_rows > 0


def _check(n_rows: int, k_dim: int) -> None:
    if not supported_shape(n_rows, k_dim):
        raise ValueError(f"moe_gemv: unsupported shape n_rows={n_rows} k_dim={k_dim}")


# 槽位编号，不是指针。显存里是一块 (slots x blob_bytes) 的 arena，kernel 用
# slot_id * blob_bytes 取这个专家的字节。
#
# 每层送一行编号（各层专家组合不同），所以按「层」持有一对 pinned host / device
# 缓冲，每次覆盖写。不能按 (代次, 组合) 缓存再发新 tensor：那样每层每 token 都是
# 一次新的 pageable H2D（同步），40 层就是一 token 四十次全设备停顿；而用一个只有
# 十几个槽的共享环更糟 —— 每层都在写，环很快被占满，复制会在 DMA 还没完成时被
# 下一层覆盖（这正是老指针数组注释里记的那个「decode 塌成一个重复 token」）。
# 按层复用意味着同一份 buffer 要过整层 40 层才会被写第二次，DMA 早就完成了。
_SLOT_BUF: dict = {}


def slot_ids(slots_dev, layer: int, ids) -> torch.Tensor:
    """int32 device [E]：本层 active 专家的槽位编号（覆盖写该层自己的缓冲）。"""
    ids = list(ids)
    n = len(ids)
    if n <= 0:
        return torch.empty(0, dtype=torch.int32, device=slots_dev)
    key = (slots_dev, int(layer))
    buf = _SLOT_BUF.get(key)
    # 补齐到 16 的倍数：sub-DMA 粒度的异步拷贝不是快路径（实测 48 字节的 copy
    # 要 923us，192 字节的只要 6us）。
    width = ((n + 15) // 16) * 16
    if buf is None or buf[0].numel() < width:
        if len(_SLOT_BUF) > 512:
            _SLOT_BUF.clear()
        buf = _SLOT_BUF[key] = (
            torch.empty(width, dtype=torch.int32, pin_memory=True),
            torch.empty(width, dtype=torch.int32, device=slots_dev))
    host, dev = buf
    host.numpy()[:n] = ids
    dev[:width].copy_(host, non_blocking=True)
    return dev[:n]


def _e8m0_bytes(s: torch.Tensor) -> torch.Tensor:
    """Raw e8m0 scale bytes, without the fp32 widening the tilelang path needs."""
    if s.dtype == torch.uint8:
        return s
    return s.view(torch.uint8)


def gate_up(aq: torch.Tensor, asc: torch.Tensor, arena: torch.Tensor,
            slots: torch.Tensor, blob_bytes: int, geom: tuple[int, int],
            offs: tuple[int, int, int, int]) -> torch.Tensor:
    """[n_exp, 2*n_rows] bf16: gate rows first, then up rows, per expert.

    ``arena`` 是 (slots x blob_bytes) 的显存大 buffer，``slots`` 是每个 active
    专家的槽位编号，``offs`` 是 w1/s1/w3/s3 在 blob 里的字节偏移（同一层所有
    专家一致）。kernel 取 ``arena + slot*blob_bytes + off``，不问地址。
    """
    n_exp = int(slots.numel())
    n_rows, k_dim = int(geom[0]), int(geom[1])
    _check(n_rows, k_dim)
    out = torch.empty((n_exp, 2 * n_rows), dtype=torch.bfloat16, device=aq.device)
    w1_off, s1_off, w3_off, s3_off = (int(o) for o in offs)
    native.launch(_kernel("moe_gemv6"), ((n_rows + 15) // 16, n_exp, 2), (512,),
                  [aq, _e8m0_bytes(asc), arena, slots, out, n_rows, k_dim, n_exp,
                   0, 0, int(blob_bytes), w1_off, s1_off, w3_off, s3_off], 0)
    return out


_BOUND: dict = {}


def act(gu: torch.Tensor, w_exp: torch.Tensor, limit: float = 0.0):
    """[n_exp, n] bf16: silu(clamp(gate)) * clamp(up) * routing weight."""
    n_exp, two_n = gu.shape
    n = two_n // 2
    out = torch.empty((n_exp, n), dtype=torch.bfloat16, device=gu.device)
    if w_exp.dtype != torch.float32:
        w_exp = w_exp.float()
    # Bound launcher: this runs once per layer (40 per decode step) and the generic path
    # charges ~19 us of argument marshalling for a kernel whose work here is well under
    # that. Keyed on the geometry, which is constant for a given model shape.
    key = ("moe_act6", int(n_exp), int(n))
    fn = _BOUND.get(key)
    if fn is None:
        fn = native.BoundLaunch(_kernel("moe_act6"),
                                ((n + 255) // 256, n_exp, 1), (256,), "TOTifi")
        _BOUND[key] = fn
    fn(gu, w_exp.contiguous(), out, n, float(limit), n_exp, stream=None)
    return out


def act_quant(gu: torch.Tensor, w_exp: torch.Tensor, limit: float = 0.0):
    """([n_exp, n] fp8 e4m3, [n_exp, n//32] e8m0): silu, routing weight and the
    reference's block quantisation in one launch.

    The down projection feeds on this rather than on a bf16 activation: fp8 e4m3 has a
    3-bit mantissa, and substituting the exact bf16 value shifted the final logits by 52%
    over forty layers even with each individual matmul inside a bf16 ulp.
    """
    n_exp, two_n = gu.shape
    n = two_n // 2
    if n % 32:
        raise ValueError(f"moe_gemv: activation length {n} is not a multiple of 32")
    aq = torch.empty((n_exp, n), dtype=torch.uint8, device=gu.device)
    asc = torch.empty((n_exp, n // 32), dtype=torch.uint8, device=gu.device)
    if w_exp.dtype != torch.float32:
        w_exp = w_exp.float()
    key = ("moe_act_q6", int(n_exp), int(n))
    fn = _BOUND.get(key)
    if fn is None:
        fn = native.BoundLaunch(_kernel("moe_act_q6"),
                                (n // 32, n_exp, 1), (32,), "TOTOifi")
        _BOUND[key] = fn
    fn(gu, w_exp.contiguous(), aq, asc, n, float(limit), n_exp, stream=None)
    return aq.view(torch.float8_e4m3fn), asc.view(torch.float8_e8m0fnu)


def down(aq: torch.Tensor, asc: torch.Tensor, arena: torch.Tensor,
         slots: torch.Tensor, blob_bytes: int, geom: tuple[int, int],
         w_off: int, s_off: int) -> torch.Tensor:
    """[n_exp, 2*n_rows] bf16: one down projection per expert, plane 0 used."""
    n_exp = int(slots.numel())
    n_rows, k_dim = int(geom[0]), int(geom[1])
    _check(n_rows, k_dim)
    out = torch.empty((n_exp, 2 * n_rows), dtype=torch.bfloat16, device=aq.device)
    # grid.z 仍然是 2 个 plane，只有 plane 0 被真正消费；两个 plane 都指向同一
    # 组槽位，所以 which 的两种取值拿到的是同一段权重（之前靠指针交错实现）。
    native.launch(_kernel("moe_gemv6"), ((n_rows + 15) // 16, n_exp, 1), (512,),
                  [aq, _e8m0_bytes(asc), arena, slots, out, n_rows, k_dim, n_exp,
                   k_dim, k_dim // 32, int(blob_bytes), int(w_off), int(s_off),
                   int(w_off), int(s_off)], 0)
    return out


def acc(src: torch.Tensor, y: torch.Tensor, n_exp: int) -> None:
    """y += sum over experts of src; src is [n_exp, stride_e] inside a wider buffer."""
    n = y.numel()
    stride_e = src.stride(0)
    native.launch(_kernel("moe_acc6"), ((n + 255) // 256, 1, 1), (256,),
                  [src, y, n, n_exp, stride_e], 0)
