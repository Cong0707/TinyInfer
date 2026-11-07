"""v2: correctness against a reference dot, then bandwidth."""
import sys
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
import torch
import tinyinfer.torch_compat  # noqa: F401
from tinyinfer import native

N, K = 4096, 4096
Wq = torch.randint(0, 256, (N, K//2), dtype=torch.uint8, device="cuda")
sc = torch.full((N, K//32), 127, dtype=torch.uint8, device="cuda")  # scale 2^0 = 1
a = torch.randn(K, dtype=torch.float32, device="cuda")
asc = torch.ones(K//32, dtype=torch.float32, device="cuda")
y1 = torch.zeros(N, dtype=torch.bfloat16, device="cuda")
y2 = torch.zeros(N, dtype=torch.bfloat16, device="cuda")

def run(name, y):
    native.launch(native.kernel("gemv_fp4_e8m0", name), ((N + 15)//16,), (512,),
                  [a, asc, Wq, sc, y, N, K])

def bench(fn, n=100, warm=20):
    with torch.inference_mode():
        for _ in range(warm): fn()
        torch.cuda.synchronize()
        e0 = torch.cuda.Event(True); e1 = torch.cuda.Event(True)
        e0.record()
        for _ in range(n): fn()
        e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1)/n

# sanity: does v2 produce anything close to v1?
with torch.inference_mode():
    run("gemv_fp4_e8m0", y1)
    run("gemv_fp4_e8m0_v2", y2)
    torch.cuda.synchronize()
print("v1 nonzero:", int((y1 != 0).sum()), "| v2 nonzero:", int((y2 != 0).sum()))
print("diff amax:", float((y1.float() - y2.float()).abs().max()))
print("v1 first 3:", y1[:3].tolist(), "| v2 first 3:", y2[:3].tolist())
print()

ms1 = bench(lambda: run("gemv_fp4_e8m0", y1))
ms2 = bench(lambda: run("gemv_fp4_e8m0_v2", y2))
by = N * K * 0.5
print(f"v1: {ms1*1000:7.1f} us, {by/(ms1/1000)/1e9:6.0f} GB/s")
print(f"v2: {ms2*1000:7.1f} us, {by/(ms2/1000)/1e9:6.0f} GB/s  x{ms1/ms2:.2f}")
