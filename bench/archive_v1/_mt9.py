import sys
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
import torch
import tinyinfer.torch_compat  # noqa
import tinyinfer.native as N
from pathlib import Path

orig = N._DIR
N._DIR = Path(r"E:\Develop\Deepseek\deepseek\_marlin_src")
try:
    M, Nn, K = 1, 16384, 5120
    b_words = (K // 16) * (Nn // 2) * 4
    a_words = K * 2 // 4
    c_words = Nn * 2 // 4
    s_words = 8704 * 2

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

    # 35840*2 bytes needed; must raise the dynamic-smem cap above 48KB? 35840 < 48K so no,
    # it just needs to be >= the kernel's request. Set the attribute to be safe.
    fn = N.kernel("marlin_bench",
      "_ZN6marlin6MarlinILx1125899906910725ELx1125899907892224ELx1125899906910725ELx1125899906910725ELi128ELi1ELi16ELi4ELb0ELi4ELin1ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_iiiiPibbbi")
    print("fn handle:", fn)

    tn = 16
    nblocks = Nn // (tn * 16)
    for shared in (40960, 49152, 65536, 98304):
        args = [A, B, C, C_tmp, bias, a_scales, S, gscale, zp,
                M, Nn, K, K, locks, 0, 0, 0, shared]
        try:
            N.launch(fn, (nblocks,), (128,), args, shared=shared)
            torch.cuda.synchronize()
            with torch.inference_mode():
                for _ in range(5):
                    N.launch(fn, (nblocks,), (128,), args, shared=shared)
                torch.cuda.synchronize()
                e0 = torch.cuda.Event(True); e1 = torch.cuda.Event(True)
                e0.record()
                for _ in range(30):
                    N.launch(fn, (nblocks,), (128,), args, shared=shared)
                e1.record(); torch.cuda.synchronize()
            ms = e0.elapsed_time(e1) / 30
            gb = Nn * K * 0.5 / 2**30
            print(f"shared={shared}: {ms*1000:8.1f} us -> {gb/(ms/1000):6.0f} GB/s")
            break
        except Exception as e:
            print(f"shared={shared}: {str(e)[:60]}")
finally:
    N._DIR = orig