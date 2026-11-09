"""Block-32 fp8 -> bf16 dequantisation, on our own kernel."""

import torch


def dequant_fp8(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """weight [O, I] float8_e4m3fn, scale [O/32, I/32] e8m0 bytes -> bf16."""
    from .kernels import native

    O, I = int(weight.shape[0]), int(weight.shape[1])
    if I % 32 or O % 32:
        raise ValueError(f'dequant_fp8 needs O%32==0 and I%32==0, got O={O} I={I}')
    # **进入内核前必须保证两个输入都 16 字节对齐。**
    #
    # 内核按 16 字节向量化读（权重和 scale 都是）。而真实模型里这两个张量是
    # 从 safetensors 里切出来的视图，起点不保证对齐：实测 wo_a.scale 的
    # data_ptr % 16 == 8。
    #
    # 未对齐的后果很隐蔽：CUDA 不一定报错，那条向量化指令可能被静默丢掉，
    # 输出保持 torch.empty 的初值（全 0）。调用方于是拿到一个形状正确、
    # 数值全零的「权重」，一路算下去 logits 变成常数或 NaN，而没有任何一处
    # 抛出异常。用 fp64 对拍时表现为「误差恰好等于参考值本身的量级」，
    # 很容易被误读成精度问题。
    #
    # 拷贝一次是最便宜的正确性保障：scale 只有 32KB 量级，weight 一般是连续
    # 的（contiguous 检查为 True 时 data_ptr 不变、不产生拷贝）。
    # 注意用 clone() 而不是 contiguous()：对「已经连续、但起点不对齐」的张量，
    # contiguous() 是 no-op（data_ptr 不变），照样会喂给内核一个非对齐指针。
    # clone() 一定重新分配，而 torch 的分配器返回的地址至少是 512 字节对齐的。
    if weight.data_ptr() % 16 or (not weight.is_contiguous()):
        weight = weight.clone()
    if scale.data_ptr() % 16 or (not scale.is_contiguous()):
        scale = scale.clone()
    out = torch.empty((O, I), dtype=torch.bfloat16, device=weight.device)
    # One thread per 32-element scale block; the kernel strides over blocks itself, so the
    # grid only needs enough blocks to fill the SMs.
    blocks = O * (I // 32)
    grid = (min(4096, (blocks + 255) // 256),)
    b = native.bind('dequant_fp8', 'dequant_fp8', grid, (256,), 'TTOii')
    b(weight.view(torch.uint8), scale.view(torch.uint8), out, int(O), int(I),
      stream=None)
    return out
