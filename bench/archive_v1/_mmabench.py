import sys
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
import torch
import tinyinfer.torch_compat  # noqa: F401
from tinyinfer import native

N, K = 4096, 5120
w = torch.randint(0, 256, (N, K//2), dtype=torch.uint8, device="cuda")
ws = torch.full((N, K//32), 127, dtype=torch.uint8, device="cuda")
a = torch.randn(K, dtype=torch.bfloat16, device="cuda") * 0.1
asc = torch.ones(K//32, dtype=torch.float32, device="cuda")
ab = a.float()
y = torch.zeros(N, dtype=torch.bfloat16, device="cuda")

def bench(fn, n=50, warm=10):
    with torch.inference_mode():
        for _ in range(warm): fn()
        torch.cuda.synchronize()
        e0 = torch.cuda.Event(True); e1 = torch.cuda.Event(True)
        e0.record()
        for _ in range(n): fn()
        e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1)/n

by = N*K*0.5
# scalar ALU path
ms_s = bench(lambda: native.launch(native.kernel("gemv_fp4_e8m0","gemv_fp4_fast"),
                                   ((N+7)//8,), (256,), [ab, asc, w, ws, y, N, K]))
# mma path: grid = ceil(N / (8 warps * 16 rows))
grid_mma = ((N // 128) + 1,)
try:
    native.launch(native.kernel("gemv_fp4_mma","gemv_fp4_mma"), grid_mma, (256,),
                  [a, asc, w, ws, y, N, K])
    torch.cuda.synchronize()
    ms_m = bench(lambda: native.launch(native.kernel("gemv_fp4_mma","gemv_fp4_mma"),
                                       grid_mma, (256,), [a, asc, w, ws, y, N, K]))
    print(f"scalar: {ms_s*1000:8.1f} us -> {by/(ms_s/1000)/1e9:6.0f} GB/s")
    print(f"mma   : {ms_m*1000:8.1f} us -> {by/(ms_m/1000)/1e9:6.0f} GB/s  x{ms_s/ms_m:.2f}")
except Exception as e:
    print("mma launch failed:", type(e).__name__, str(e)[:200])
