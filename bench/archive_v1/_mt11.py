import sys, ctypes
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
import torch
import tinyinfer.torch_compat  # noqa
import tinyinfer.native as N
from pathlib import Path

MANGLE = ("_ZN6marlin6MarlinILx1125899906910725ELx1125899907892224ELx1125899906910725E"
          "Lx1125899906910725ELi128ELi1ELi16ELi4ELb0ELi4ELin1ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_iiiiPibbbi")
MANGLE2 = ("_ZN6marlin6MarlinILx1125899906910725ELx1125899907892224ELx1125899906910725E"
           "Lx1125899906910725ELi128ELi1ELi8ELi8ELb0ELi4ELin1ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_iiiiPibbbi")

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

    sms = 26
    gb = Nn * K * 0.5 / 2**30
    for mangle, label, tn in ((MANGLE, "tm1 tn16 tk4", 16), (MANGLE2, "tm1 tn8 tk8", 8)):
        fn = N.kernel("marlin_bench", mangle)
        lib.cuFuncSetAttribute(fn, 8, ctypes.c_int(101376))
        best = None
        for nblocks in (sms, sms*2, sms*4, Nn // (tn*16), Nn // (tn*16)*2):
            args = [A, B, C, C_tmp, bias, a_scales, S, gscale, zp,
                    M, Nn, K, K, locks, 0, 0, 0, 49152]
            try:
                N.launch(fn, (nblocks,), (128,), args, shared=49152)
                torch.cuda.synchronize()
                with torch.inference_mode():
                    for _ in range(5):
                        N.launch(fn, (nblocks,), (128,), args, shared=49152)
                    torch.cuda.synchronize()
                    e0 = torch.cuda.Event(True); e1 = torch.cuda.Event(True)
                    e0.record()
                    for _ in range(30):
                        N.launch(fn, (nblocks,), (128,), args, shared=49152)
                    e1.record(); torch.cuda.synchronize()
                ms = e0.elapsed_time(e1)/30
                g = gb/(ms/1000)
                if best is None or g > best[0]:
                    best = (g, ms, nblocks)
            except Exception:
                pass
        if best:
            print(f"{label}: best {best[1]*1000:8.1f} us -> {best[0]:6.0f} GB/s (blocks={best[2]})")
        else:
            print(f"{label}: all failed")
    print()
    print("mine gemv_fp4_fast: 68 GB/s")
finally:
    N._DIR = orig
