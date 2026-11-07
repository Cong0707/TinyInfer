import sys
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
import torch
import tinyinfer.torch_compat  # noqa
from tinyinfer import native

def bench(fn, n=25, warm=6):
    with torch.inference_mode():
        for _ in range(warm): fn()
        torch.cuda.synchronize()
        e0 = torch.cuda.Event(True); e1 = torch.cuda.Event(True)
        e0.record()
        for _ in range(n): fn()
        e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1)/n

N, K = 16384, 5120
w = torch.randint(0, 2**31 - 1, (N, K//8), dtype=torch.int32, device="cuda")
ws = torch.full((N, K//32), 127, dtype=torch.uint8, device="cuda")
x = torch.randn(K, dtype=torch.bfloat16, device="cuda")
y = torch.zeros(N, dtype=torch.bfloat16, device="cuda")
by = N*K*0.5
# 8 warps -> 128 rows per block, 256 threads
grid8 = ((N + 127)//128,)
try:
    native.launch(native.kernel("marlin_fp4","e2m1_v4"), grid8, (256,), [x, w, ws, y, N, K])
    torch.cuda.synchronize()
    ms = bench(lambda: native.launch(native.kernel("marlin_fp4","e2m1_v4"), grid8, (256,),
                                     [x, w, ws, y, N, K]))
    print(f"8 warps: {ms*1000:8.1f} us -> {by/(ms/1000)/1e9:6.0f} GB/s  (was 19)")
except Exception as e:
    print("failed:", type(e).__name__, str(e)[:200])
