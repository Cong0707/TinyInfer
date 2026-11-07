import sys
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
import torch
import tinyinfer.torch_compat  # noqa: F401
from tinyinfer import native

N, K = 4096, 4096
Wq = torch.randint(0, 256, (N, K//2), dtype=torch.uint8, device="cuda")
sc = torch.full((N, K//32), 127, dtype=torch.uint8, device="cuda")
a = torch.randn(K, dtype=torch.float32, device="cuda")
asc = torch.ones(K//32, dtype=torch.float32, device="cuda")
y1 = torch.zeros(N, dtype=torch.bfloat16, device="cuda")
y3 = torch.zeros(N, dtype=torch.bfloat16, device="cuda")

def bench(fn, n=100, warm=20):
    with torch.inference_mode():
        for _ in range(warm): fn()
        torch.cuda.synchronize()
        e0 = torch.cuda.Event(True); e1 = torch.cuda.Event(True)
        e0.record()
        for _ in range(n): fn()
        e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1)/n

with torch.inference_mode():
    native.launch(native.kernel("gemv_fp4_e8m0","gemv_fp4_e8m0"),
                  ((N+15)//16,), (512,), [a, asc, Wq, sc, y1, N, K])
    try:
        native.launch(native.kernel("gemv_fp4_e8m0","gemv_fp4_e8m0_v3"),
                      ((N+7)//8,), (256,), [a, asc, Wq, sc, y3, N, K], shared=K*4)
    except Exception as e:
        print("v3 launch failed:", e)
        sys.exit(1)
    torch.cuda.synchronize()
print("diff amax:", float((y1.float()-y3.float()).abs().max()))
print("v1[0:3]:", y1[:3].tolist(), "v3[0:3]:", y3[:3].tolist())
by = N*K*0.5
ms1 = bench(lambda: native.launch(native.kernel("gemv_fp4_e8m0","gemv_fp4_e8m0"),
                                  ((N+15)//16,), (512,), [a,asc,Wq,sc,y1,N,K]))
ms3 = bench(lambda: native.launch(native.kernel("gemv_fp4_e8m0","gemv_fp4_e8m0_v3"),
                                  ((N+7)//8,), (256,), [a,asc,Wq,sc,y3,N,K], shared=K*4))
print(f"v1: {ms1*1000:7.1f} us, {by/(ms1/1000)/1e9:6.0f} GB/s")
print(f"v3: {ms3*1000:7.1f} us, {by/(ms3/1000)/1e9:6.0f} GB/s  x{ms1/ms3:.2f}")
