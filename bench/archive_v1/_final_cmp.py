import sys, ctypes
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
import torch
import tinyinfer.torch_compat  # noqa
import tinyinfer.native as N
from pathlib import Path

MANGLE = ("_ZN6marlin6MarlinILx1125899906910725ELx1125899907892224ELx1125899906910725E"
          "Lx1125899906910725ELi128ELi1ELi16ELi4ELb0ELi4ELin1ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_iiiiPibbbi")

# load the project kernel first, while _DIR still points at the project
_mine_fn = N.kernel("gemv_fp4_e8m0", "gemv_fp4_fast")

orig = N._DIR
N._DIR = Path(r"E:\Develop\Deepseek\deepseek\_marlin_src")
try:
    lib = N._lib()
    M, Nn, K = 1, 16384, 5120
    b_words = (K // 16) * (Nn // 2) * 4
    A = torch.randint(0, 2**15, (K,), dtype=torch.int16, device="cuda").view(torch.int32)
    B = torch.randint(0, 2**15, (b_words * 2,), dtype=torch.int16, device="cuda").view(torch.int32)
    C = torch.zeros(Nn * 2 // 4, dtype=torch.int32, device="cuda")
    C_tmp = torch.zeros(max(64, M * Nn * 2), dtype=torch.float32, device="cuda")
    S = torch.zeros(8704 * 2, dtype=torch.int32, device="cuda")
    locks = torch.zeros(8192, dtype=torch.int32, device="cuda")
    a_scales = torch.ones(1, dtype=torch.float32, device="cuda")
    gscale = torch.ones(1, dtype=torch.float32, device="cuda")
    zp = torch.zeros(64, dtype=torch.int32, device="cuda")
    bias = torch.zeros(64, dtype=torch.int32, device="cuda")

    fn = N.kernel("marlin_bench", MANGLE)
    lib.cuFuncSetAttribute(fn, 8, ctypes.c_int(101376))
    args = [A, B, C, C_tmp, bias, a_scales, S, gscale, zp,
            M, Nn, K, K, locks, 0, 0, 0, 49152]
    gb = Nn * K * 0.5 / 2**30

    def bench(n=50, warm=10):
        with torch.inference_mode():
            for _ in range(warm):
                N.launch(fn, (26,), (128,), args, shared=49152)
            torch.cuda.synchronize()
            e0 = torch.cuda.Event(True); e1 = torch.cuda.Event(True)
            e0.record()
            for _ in range(n):
                N.launch(fn, (26,), (128,), args, shared=49152)
            e1.record(); torch.cuda.synchronize()
        return e0.elapsed_time(e1)/n

    print("=== 预热 3 轮后测 ===")
    for rnd in range(3):
        ms = bench()
        print(f"  round {rnd}: {ms*1000:8.1f} us -> {gb/(ms/1000):6.0f} GB/s")
    print()
    print("=== 我的 kernel 同尺寸同条件 ===")
    mine_fn = _mine_fn
    w8 = torch.randint(0, 2**31-1, (Nn, K//8), dtype=torch.int32, device="cuda").view(torch.uint8)
    ws8 = torch.full((Nn, K//32), 127, dtype=torch.uint8, device="cuda")
    xf = torch.randn(K, dtype=torch.float32, device="cuda")
    asc = torch.ones(K//32, dtype=torch.float32, device="cuda")
    y = torch.zeros(Nn, dtype=torch.bfloat16, device="cuda")
    def bench_mine(n=50, warm=10):
        with torch.inference_mode():
            for _ in range(warm):
                N.launch(mine_fn, ((Nn+7)//8,), (256,), [xf, asc, w8, ws8, y, Nn, K])
            torch.cuda.synchronize()
            e0 = torch.cuda.Event(True); e1 = torch.cuda.Event(True)
            e0.record()
            for _ in range(n):
                N.launch(mine_fn, ((Nn+7)//8,), (256,), [xf, asc, w8, ws8, y, Nn, K])
            e1.record(); torch.cuda.synchronize()
        return e0.elapsed_time(e1)/n
    ms_m = bench_mine()
    print(f"  mine: {ms_m*1000:8.1f} us -> {gb/(ms_m/1000):6.0f} GB/s")
    ms_v = bench()
    print()
    print(f"vLLM Marlin: {gb/(ms_v/1000):6.0f} GB/s")
    print(f"mine       : {gb/(ms_m/1000):6.0f} GB/s")
    print(f"vLLM / mine = {(gb/(ms_v/1000))/(gb/(ms_m/1000)):.2f}x")
finally:
    N._DIR = orig