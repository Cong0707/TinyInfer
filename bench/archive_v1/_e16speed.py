import sys
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
import torch
import tinyinfer.torch_compat  # noqa
from tinyinfer import native

N, K = 16384, 5120
w = torch.randint(0, 2**31 - 1, (N, K//8), dtype=torch.int32, device="cuda")
ws = torch.full((N, K//32), 127, dtype=torch.uint8, device="cuda")
x = torch.randn(K, dtype=torch.bfloat16, device="cuda")
y = torch.zeros(N, dtype=torch.bfloat16, device="cuda")
asc = torch.ones(K//32, dtype=torch.float32, device="cuda")

def bench(fn, n=30, warm=8):
    with torch.inference_mode():
        for _ in range(warm): fn()
        torch.cuda.synchronize()
        e0 = torch.cuda.Event(True); e1 = torch.cuda.Event(True)
        e0.record()
        for _ in range(n): fn()
        e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1)/n

grid = ((N + 63)//64,)
ms = bench(lambda: native.launch(native.kernel("marlin_fp4","e2m1_gemv16"), grid, (128,),
                                 [x, w, ws, y, N, K]))
by = N*K*0.5
print(f"e2m1_gemv16: {ms*1000:8.1f} us -> {by/(ms/1000)/1e9:6.0f} GB/s")
print()
print("对比：")
print("  tilelang (引擎现用)  41 GB/s")
print("  我的标量 ALU 版      62 GB/s")
print("  纯读取上限          212 GB/s")
