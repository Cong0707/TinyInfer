import sys
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
import torch
import tinyinfer.torch_compat  # noqa
import tinyinfer.native as N
from pathlib import Path

# thread_n=16, thread_k=4 -> thread_k = 4*16 = 64 real k per block
MANGLES = {
  "tm1 tn16 tk4 g-1": "_ZN6marlin6MarlinILx1125899906910725ELx1125899907892224ELx1125899906910725ELx1125899906910725ELi128ELi1ELi16ELi4ELb0ELi4ELin1ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_iiiiPibbbi",
  "tm1 tn8 tk8 g-1":  "_ZN6marlin6MarlinILx1125899906910725ELx1125899907892224ELx1125899906910725ELx1125899906910725ELi128ELi1ELi8ELi8ELb0ELi4ELin1ELb0EEEvPK4int4S3_PS1_S4_S3_PKfS3_S6_S3_iiiiPibbbi",
}

orig = N._DIR
N._DIR = Path(r"E:\Develop\Deepseek\deepseek\_marlin_src")
try:
    M, Nn, K = 1, 1024, 2048   # thread_k=64 -> K%64==0 OK; thread_n=256 -> N%256==0
    for label, m in MANGLES.items():
        # generous buffers
        A = torch.zeros(max(64, M * K // 8), dtype=torch.int32, device="cuda")
        B = torch.zeros(max(64, K * Nn // 8) * 2, dtype=torch.int32, device="cuda")
        C = torch.zeros(max(64, M * Nn // 8), dtype=torch.int32, device="cuda")
        C_tmp = torch.zeros(max(64, M * Nn * 4), dtype=torch.float32, device="cuda")
        S = torch.zeros(max(64, (K // 128) * Nn) * 2, dtype=torch.int32, device="cuda")
        locks = torch.zeros(8192, dtype=torch.int32, device="cuda")
        a_scales = torch.zeros(1, dtype=torch.float32, device="cuda")
        gscale = torch.zeros(1, dtype=torch.float32, device="cuda")
        zp = torch.zeros(256, dtype=torch.int32, device="cuda")
        bias = torch.zeros(256, dtype=torch.int32, device="cuda")
        tn = 16 if "tn16" in label else 8
        nblocks = max(1, Nn // (tn * 16))
        args = [A, B, C, C_tmp, bias, a_scales, S, gscale, zp,
                M, Nn, K, K, locks, 0, 0, 0, 32768]
        try:
            N.launch(N.kernel("marlin_bench", m), (nblocks,), (128,), args, shared=32768)
            torch.cuda.synchronize()
            with torch.inference_mode():
                for _ in range(3):
                    N.launch(N.kernel("marlin_bench", m), (nblocks,), (128,), args, shared=32768)
                torch.cuda.synchronize()
                e0 = torch.cuda.Event(True); e1 = torch.cuda.Event(True)
                e0.record()
                for _ in range(30):
                    N.launch(N.kernel("marlin_bench", m), (nblocks,), (128,), args, shared=32768)
                e1.record(); torch.cuda.synchronize()
            ms = e0.elapsed_time(e1)/30
            gb = Nn * K * 0.5 / 2**30
            print(f"{label}: {ms*1000:8.1f} us -> {gb/(ms/1000):6.0f} GB/s (blocks={nblocks})")
        except Exception as e:
            print(f"{label}: {type(e).__name__} {str(e)[:70]}")
            break
finally:
    N._DIR = orig
