import sys
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
    M, Nn, K = 1, 16384, 5120
    A = torch.randint(0, 2**15, (max(1, M * K // 8),), dtype=torch.int16, device="cuda").view(torch.int32)
    B = torch.randint(0, 2**31-1, (K//8 * Nn // 4,), dtype=torch.int32, device="cuda")
    C = torch.zeros(max(1, M * Nn // 8), dtype=torch.int32, device="cuda")
    C_tmp = torch.zeros(max(1, M * Nn), dtype=torch.float32, device="cuda")
    scales = torch.ones(max(8, (K//128) * Nn // 8), dtype=torch.int32, device="cuda")
    locks = torch.zeros(4096, dtype=torch.int32, device="cuda")
    a_scales = torch.ones(1, dtype=torch.float32, device="cuda")
    gscale = torch.ones(1, dtype=torch.float32, device="cuda")
    zp = torch.zeros(8, dtype=torch.int32, device="cuda")
    bias = torch.zeros(8, dtype=torch.int32, device="cuda")

    sms = 26
    args = [A.data_ptr(), B.data_ptr(), C.data_ptr(), C_tmp.data_ptr(), bias.data_ptr(),
            a_scales.data_ptr(), scales.data_ptr(), gscale.data_ptr(), zp.data_ptr(),
            M, Nn, K, K, locks.data_ptr(), False, False, False, 101376]

    def bench(name, grid, block, shared):
        N.launch(N.kernel("marlin_bench", name), grid, block, args, shared=shared)
        torch.cuda.synchronize()
        with torch.inference_mode():
            for _ in range(5):
                N.launch(N.kernel("marlin_bench", name), grid, block, args, shared=shared)
            torch.cuda.synchronize()
            e0 = torch.cuda.Event(True); e1 = torch.cuda.Event(True)
            e0.record()
            for _ in range(20):
                N.launch(N.kernel("marlin_bench", name), grid, block, args, shared=shared)
            e1.record(); torch.cuda.synchronize()
        return e0.elapsed_time(e1)/20

    gb = Nn * K * 0.5 / 2**30
    # thread_n=16 blocks -> grid = N/(16*16); thread_n=8 -> N/(8*16)
    for name, label, nblocks in ((MANGLE, "tm1 tn16 tk4", Nn // (16*16)),
                                 (MANGLE2, "tm1 tn8 tk8", Nn // (8*16))):
        try:
            ms = bench(name, (nblocks,), (128,), 101376)
            print(f"{label}: {ms*1000:8.1f} us -> {gb/(ms/1000):6.0f} GB/s  (blocks={nblocks})")
        except Exception as e:
            print(f"{label}: {type(e).__name__} {str(e)[:150]}")
finally:
    N._DIR = orig
