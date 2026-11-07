import sys
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
import torch
import tinyinfer.torch_compat  # noqa
import tinyinfer.native as N
from pathlib import Path

MANGLE16 = ("_ZN6marlin6MarlinILx1125899906910725ELx1125899907892224ELx1125899906910725E"
            "Lx1125899906910725ELi128ELi1ELi16ELi4ELb0ELi4ELin1ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_iiiiPibbbi")

orig = N._DIR
N._DIR = Path(r"E:\Develop\Deepseek\deepseek\_marlin_src")
try:
    # start small so every buffer is obviously big enough, then scale
    for (M, Nn, K) in ((1, 256, 512), (1, 1024, 2048), (1, 16384, 5120)):
        # A: fp16 [M, K]  -> int4 words
        a_words = max(1, (M * K * 2 + 15) // 16)
        # B: marlin packs [K/16, N] into int4 words; generous 4x
        b_words = max(1, (K * Nn // 8)) * 2
        # C: fp16 [M, N]
        c_words = max(1, (M * Nn * 2 + 15) // 16)
        # C_tmp: fp32 [M, N]
        ctmp = max(1, M * Nn)
        # scales: fp16 [K/128, N] -> int4 words
        s_words = max(1, (K // 128) * Nn * 2 // 16 + 8) * 2
        locks_n = 4096

        A = torch.zeros(a_words, dtype=torch.int32, device="cuda")
        B = torch.zeros(b_words, dtype=torch.int32, device="cuda")
        C = torch.zeros(c_words, dtype=torch.int32, device="cuda")
        C_tmp = torch.zeros(ctmp, dtype=torch.float32, device="cuda")
        S = torch.zeros(s_words, dtype=torch.int32, device="cuda")
        locks = torch.zeros(locks_n, dtype=torch.int32, device="cuda")
        a_scales = torch.zeros(1, dtype=torch.float32, device="cuda")
        gscale = torch.zeros(1, dtype=torch.float32, device="cuda")
        zp = torch.zeros(64, dtype=torch.int32, device="cuda")
        bias = torch.zeros(64, dtype=torch.int32, device="cuda")

        nblocks = max(1, Nn // (16 * 16))
        args = [A, B, C, C_tmp, bias, a_scales, S, gscale, zp,
                M, Nn, K, K, locks, 0, 0, 0, 32768]
        try:
            N.launch(N.kernel("marlin_bench", MANGLE16), (nblocks,), (128,), args, shared=32768)
            torch.cuda.synchronize()
            with torch.inference_mode():
                for _ in range(3):
                    N.launch(N.kernel("marlin_bench", MANGLE16), (nblocks,), (128,), args, shared=32768)
                torch.cuda.synchronize()
                e0 = torch.cuda.Event(True); e1 = torch.cuda.Event(True)
                e0.record()
                for _ in range(20):
                    N.launch(N.kernel("marlin_bench", MANGLE16), (nblocks,), (128,), args, shared=32768)
                e1.record(); torch.cuda.synchronize()
            ms = e0.elapsed_time(e1) / 20
            gb = Nn * K * 0.5 / 2**30
            print(f"M={M} N={Nn} K={K}: {ms*1000:8.1f} us -> {gb/(ms/1000):6.0f} GB/s (blocks={nblocks})")
        except Exception as e:
            print(f"M={M} N={Nn} K={K}: {type(e).__name__} {str(e)[:80]}")
            break
finally:
    N._DIR = orig
