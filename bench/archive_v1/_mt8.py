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
    # B: exactly K/16 * (N/2) int4 words = 2621440*4 int32
    b_words = (K // 16) * (Nn // 2) * 4
    # A: fp16 [1, K] = K*2 bytes = 10240 bytes -> 2560 int32
    a_words = K * 2 // 4
    # C: fp16 [1, N] = N*2 bytes -> N/2 int32
    c_words = Nn * 2 // 4
    # scales: fp16 [K/group, N]; group=-1 means one scale -> small
    # group_blocks=-1: s_gl_rd = s_sh_stride*slice_col + tid, s_sh_stride=16*tn/8=32
    # slices = N/(tn*16) = 64 -> max index 32*64+128 = 2176 int4 = 8704 int32
    s_words = 8704 * 2
    print(f"B {b_words*4/2**20:.1f} MB, A {a_words*4/2**10:.1f} KB, C {c_words*4/2**10:.1f} KB")

    A = torch.zeros(a_words, dtype=torch.int32, device="cuda")
    B = torch.zeros(b_words, dtype=torch.int32, device="cuda")
    C = torch.zeros(c_words, dtype=torch.int32, device="cuda")
    C_tmp = torch.zeros(max(64, M * Nn * 2), dtype=torch.float32, device="cuda")
    S = torch.zeros(s_words, dtype=torch.int32, device="cuda")
    locks = torch.zeros(8192, dtype=torch.int32, device="cuda")
    a_scales = torch.zeros(1, dtype=torch.float32, device="cuda")
    gscale = torch.zeros(1, dtype=torch.float32, device="cuda")
    zp = torch.zeros(64, dtype=torch.int32, device="cuda")
    bias = torch.zeros(64, dtype=torch.int32, device="cuda")

    tn = 16
    nblocks = Nn // (tn * 16)
    args = [A, B, C, C_tmp, bias, a_scales, S, gscale, zp,
            M, Nn, K, K, locks, 0, 0, 0, 32768]
    N.launch(N.kernel("marlin_bench", MANGLE), (nblocks,), (128,), args, shared=32768)
    torch.cuda.synchronize()
    print("launched OK")
    with torch.inference_mode():
        for _ in range(5):
            N.launch(N.kernel("marlin_bench", MANGLE), (nblocks,), (128,), args, shared=32768)
        torch.cuda.synchronize()
        e0 = torch.cuda.Event(True); e1 = torch.cuda.Event(True)
        e0.record()
        for _ in range(30):
            N.launch(N.kernel("marlin_bench", MANGLE), (nblocks,), (128,), args, shared=32768)
        e1.record(); torch.cuda.synchronize()
    ms = e0.elapsed_time(e1) / 30
    gb = Nn * K * 0.5 / 2**30
    print(f"vLLM Marlin: {ms*1000:8.1f} us -> {gb/(ms/1000):6.0f} GB/s")
    print(f"  mine gemv_fp4_fast:  68 GB/s")
    print(f"  ratio: {68/(gb/(ms/1000)):.2f}x")
except Exception as e:
    print("failed:", type(e).__name__, str(e)[:300])
finally:
    N._DIR = orig
