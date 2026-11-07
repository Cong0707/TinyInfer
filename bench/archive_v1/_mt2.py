import sys
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
import torch
import tinyinfer.torch_compat  # noqa
import tinyinfer.native as N
from pathlib import Path

orig = N._DIR
N._DIR = Path(r"E:\Develop\Deepseek\deepseek\_marlin_src")
try:
    # M=1, N=16384, K=5120 -- same shape as my kernel benchmark
    M, Nn, K = 1, 16384, 5120
    # Marlin: A is fp16 [M,K] int4-packed view; B is [K/8, N] int4 packed
    A = torch.randn(M * K, dtype=torch.float16, device="cuda")
    B = torch.randint(0, 2**31-1, (max(1, K//8) * Nn // 4,), dtype=torch.int32, device="cuda")
    C = torch.zeros(M * Nn, dtype=torch.float16, device="cuda")
    C_tmp = torch.zeros(max(1, M * Nn), dtype=torch.float32, device="cuda")
    scales = torch.ones(max(1, (K//128) * Nn // 8), dtype=torch.float16, device="cuda")
    locks = torch.zeros(4096, dtype=torch.int32, device="cuda")
    a_scales = torch.ones(1, dtype=torch.float32, device="cuda")
    gscale = torch.ones(1, dtype=torch.float32, device="cuda")
    zp = torch.zeros(8, dtype=torch.int32, device="cuda")
    bias = torch.zeros(8, dtype=torch.int32, device="cuda")

    sms = torch.cuda.get_device_properties(0).multi_processor_count
    print("SMs:", sms, "| shape:", M, Nn, K)
    # 51200 bytes shared mem for marlin
    ret = N.launch(N.kernel("marlin_bench", "marlin_test_launch"), (1,), (1,),
                   [A.data_ptr(), B.data_ptr(), C.data_ptr(), C_tmp.data_ptr(),
                    bias.data_ptr(), a_scales.data_ptr(), scales.data_ptr(),
                    gscale.data_ptr(), zp.data_ptr(),
                    M, Nn, K, K, locks.data_ptr(), False, False, False,
                    101376, torch.cuda.current_stream().cuda_stream, sms],
                   shared=0)
    torch.cuda.synchronize()
    print("launch ok")
except Exception as e:
    print("failed:", type(e).__name__, str(e)[:300])
finally:
    N._DIR = orig
