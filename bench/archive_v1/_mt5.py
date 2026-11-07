import sys
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
import torch
import tinyinfer.torch_compat  # noqa
import tinyinfer.native as N
from pathlib import Path

MANGLE = ("_ZN6marlin6MarlinILx1125899906910725ELx1125899907892224ELx1125899906910725E"
          "Lx1125899906910725ELi128ELi1ELi16ELi4ELb0ELi4ELin1ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_iiiiPibbbi")

orig = N._DIR
N._DIR = Path(r"E:\Develop\Deepseek\deepseek\_marlin_src")
try:
    M, Nn, K = 1, 16384, 5120
    # Marlin wants B as [K/16, N] int4 in its repacked layout; allocate generously so the
    # kernel's reads stay in bounds even without a real repack.
    # bytes needed, padded 4x for its interleaved access pattern
    b_words = (K * Nn // 8) * 4
    A = torch.zeros(max(1024, M * K // 8), dtype=torch.int32, device="cuda")
    B = torch.zeros(b_words, dtype=torch.int32, device="cuda")
    C = torch.zeros(max(1024, M * Nn // 8), dtype=torch.int32, device="cuda")
    C_tmp = torch.zeros(max(1024, M * Nn * 2), dtype=torch.float32, device="cuda")
    scales = torch.zeros(max(1024, (K // 128) * Nn // 8 * 4), dtype=torch.int32, device="cuda")
    locks = torch.zeros(4096, dtype=torch.int32, device="cuda")
    a_scales = torch.zeros(1, dtype=torch.float32, device="cuda")
    gscale = torch.zeros(1, dtype=torch.float32, device="cuda")
    zp = torch.zeros(64, dtype=torch.int32, device="cuda")
    bias = torch.zeros(64, dtype=torch.int32, device="cuda")

    args = [A, B, C, C_tmp, bias, a_scales, scales, gscale, zp,
            M, Nn, K, K, locks, 0, 0, 0, 32768]
    nblocks = Nn // (16*16)
    N.launch(N.kernel("marlin_bench", MANGLE), (nblocks,), (128,), args, shared=32768)
    torch.cuda.synchronize()
    print("launched OK, nblocks =", nblocks)

    def bench(n=20, warm=5):
        with torch.inference_mode():
            for _ in range(warm):
                N.launch(N.kernel("marlin_bench", MANGLE), (nblocks,), (128,), args, shared=32768)
            torch.cuda.synchronize()
            e0 = torch.cuda.Event(True); e1 = torch.cuda.Event(True)
            e0.record()
            for _ in range(n):
                N.launch(N.kernel("marlin_bench", MANGLE), (nblocks,), (128,), args, shared=32768)
            e1.record(); torch.cuda.synchronize()
        return e0.elapsed_time(e1)/n
    ms = bench()
    gb = Nn * K * 0.5 / 2**30
    print(f"vLLM Marlin (tm1 tn16 tk4): {ms*1000:8.1f} us -> {gb/(ms/1000):6.0f} GB/s")
    print(f"  (my gemv_fp4_fast: 68 GB/s, tilelang: 40 GB/s)")
except Exception as e:
    print("failed:", type(e).__name__, str(e)[:300])
finally:
    N._DIR = orig
