"""Which of my kernels is actually fastest AND correct? Full comparison."""
import sys
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
import torch
import tinyinfer.torch_compat  # noqa
from tinyinfer import native, gemv_fp4

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
words = torch.randint(0, 2**31 - 1, (N, K//8), dtype=torch.int64)
w_i32 = words.to(torch.int32).to("cuda")
w_u8 = w_i32.view(torch.uint8)
ws_u8 = torch.full((N, K//32), 127, dtype=torch.uint8, device="cuda")
x = torch.randn(K, dtype=torch.bfloat16, device="cuda")
xf = x.float()
asc = torch.ones(K//32, dtype=torch.float32, device="cuda")
y = torch.zeros(N, dtype=torch.bfloat16, device="cuda")
by = N*K*0.5

results = []
# 1. scalar ALU (fast) -- correct E2M1?  it decodes bits as E2M1 exponent form
def run_scalar():
    native.launch(native.kernel("gemv_fp4_e8m0","gemv_fp4_fast"), ((N+7)//8,), (256,),
                  [xf, asc, w_u8, ws_u8, y, N, K])
ms = bench(run_scalar)
results.append(("scalar fast", ms, by/(ms/1000)/1e9))

# 2. e2m1_v4 tensor core
def run_v4():
    native.launch(native.kernel("marlin_fp4","e2m1_v4"), ((N+127)//128,), (256,),
                  [x, w_i32, ws_u8, y, N, K])
ms = bench(run_v4)
results.append(("e2m1_v4 mma", ms, by/(ms/1000)/1e9))

# 3. tilelang (engine's current)
def run_tl():
    gemv_fp4.gemv_fp4(x, w_u8, ws_u8)
ms = bench(run_tl)
results.append(("tilelang", ms, by/(ms/1000)/1e9))

print(f"{'kernel':16s} {'us':>9s} {'GB/s':>7s}")
for n, ms, g in results:
    print(f"{n:16s} {ms*1000:9.1f} {g:7.0f}")
print()
print("pure read ceiling: 212 GB/s")
