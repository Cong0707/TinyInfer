import sys
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
import torch
import tinyinfer.torch_compat  # noqa
import tinyinfer.native as N
from pathlib import Path

orig = N._DIR
N._DIR = Path(r"E:\Develop\Deepseek\deepseek\_marlin_src")
try:
    MANGLE = ("_ZN6marlin6MarlinILx1125899906910725ELx1125899907892224ELx1125899906910725E"
              "Lx1125899906910725ELi128ELi1ELi16ELi4ELb0ELi4ELin1ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_iiiiPibbbi")
    M, Nn, K = 1, 16384, 5120
    A = torch.zeros(max(1, M * K // 8), dtype=torch.int32, device="cuda")
    B = torch.zeros(max(1, K//8 * Nn // 4), dtype=torch.int32, device="cuda")
    C = torch.zeros(max(1, M * Nn // 8), dtype=torch.int32, device="cuda")
    C_tmp = torch.zeros(max(1, M * Nn), dtype=torch.float32, device="cuda")
    scales = torch.zeros(max(8, (K//128) * Nn // 8), dtype=torch.int32, device="cuda")
    locks = torch.zeros(4096, dtype=torch.int32, device="cuda")
    a_scales = torch.zeros(1, dtype=torch.float32, device="cuda")
    gscale = torch.zeros(1, dtype=torch.float32, device="cuda")
    zp = torch.zeros(8, dtype=torch.int32, device="cuda")
    bias = torch.zeros(8, dtype=torch.int32, device="cuda")

    # try progressively smaller shared mem and grid
    for shared in (101376, 65536, 32768, 0):
        for nblocks in (Nn // (16*16), 26):
            try:
                N.launch(N.kernel("marlin_bench", MANGLE), (nblocks,), (128,),
                         [A, B, C, C_tmp, bias, a_scales, scales, gscale, zp,
                          M, Nn, K, K, locks, 0, 0, 0, shared], shared=shared)
                torch.cuda.synchronize()
                print(f"OK  shared={shared} blocks={nblocks}")
            except Exception as e:
                print(f"FAIL shared={shared} blocks={nblocks}: {str(e)[:60]}")
finally:
    N._DIR = orig
