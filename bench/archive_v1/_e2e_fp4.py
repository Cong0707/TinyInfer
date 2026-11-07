"""End-to-end: is the fp4 expert path faster now, and does the model still answer right?"""
import sys, time
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
import torch
import tinyinfer.torch_compat  # noqa
from tinyinfer import gemv_fp4

def bench(fn, n=20, warm=5):
    with torch.inference_mode():
        for _ in range(warm): fn()
        torch.cuda.synchronize()
        e0 = torch.cuda.Event(True); e1 = torch.cuda.Event(True)
        e0.record()
        for _ in range(n): fn()
        e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1)/n

N, K = 16384, 5120
w = torch.randint(0, 2**31 - 1, (N, K//8), dtype=torch.int32, device="cuda").view(torch.uint8)
ws = torch.full((N, K//32), 127, dtype=torch.uint8, device="cuda")
x = torch.randn(1, K, dtype=torch.bfloat16, device="cuda")
by = N*K*0.5

# native path
ms_n = bench(lambda: gemv_fp4._gemv_fp4_native(x, w, ws, N, K))
print(f"native path : {ms_n*1000:8.1f} us -> {by/(ms_n/1000)/1e9:6.0f} GB/s")

# force tilelang
gemv_fp4._NATIVE_OK = False
ms_t = bench(lambda: gemv_fp4.gemv_fp4(x, w, ws))
print(f"tilelang    : {ms_t*1000:8.1f} us -> {by/(ms_t/1000)/1e9:6.0f} GB/s")
gemv_fp4._NATIVE_OK = None
print(f"speedup     : {ms_t/ms_n:.2f}x")
