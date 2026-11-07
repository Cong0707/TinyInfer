import sys
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
import torch
import tinyinfer.torch_compat  # noqa: F401
from tinyinfer import native

def bench(name, grid, block, args, shared=0, n=80):
    with torch.inference_mode():
        for _ in range(15):
            native.launch(native.kernel("gemv_fp4_e8m0", name), grid, block, args, shared=shared)
        torch.cuda.synchronize()
        e0 = torch.cuda.Event(True); e1 = torch.cuda.Event(True)
        e0.record()
        for _ in range(n):
            native.launch(native.kernel("gemv_fp4_e8m0", name), grid, block, args, shared=shared)
        e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1)/n

N, K = 4096, 4096
Wq = torch.randint(0, 256, (N, K//2), dtype=torch.uint8, device="cuda")
sc = torch.full((N, K//32), 127, dtype=torch.uint8, device="cuda")
a = torch.randn(K, dtype=torch.float32, device="cuda")
asc = torch.ones(K//32, dtype=torch.float32, device="cuda")
y1 = torch.zeros(N, dtype=torch.bfloat16, device="cuda")
y4 = torch.zeros(N, dtype=torch.bfloat16, device="cuda")
by = N*K*0.5

with torch.inference_mode():
    native.launch(native.kernel("gemv_fp4_e8m0","gemv_fp4_e8m0"),
                  ((N+15)//16,), (512,), [a,asc,Wq,sc,y1,N,K])
    try:
        native.launch(native.kernel("gemv_fp4_e8m0","gemv_fp4_e8m0_v4"),
                      ((N+7)//8,), (256,), [a,asc,Wq,sc,y4,N,K])
        torch.cuda.synchronize()
        print("v4 diff amax:", float((y1.float()-y4.float()).abs().max()))
        ms4 = bench("gemv_fp4_e8m0_v4", ((N+7)//8,), (256,), [a,asc,Wq,sc,y4,N,K])
        print(f"v4: {ms4*1000:7.1f} us, {by/(ms4/1000)/1e9:6.0f} GB/s")
    except Exception as e:
        print("v4 failed:", e)
ms1 = bench("gemv_fp4_e8m0", ((N+15)//16,), (512,), [a,asc,Wq,sc,y1,N,K])
print(f"v1: {ms1*1000:7.1f} us, {by/(ms1/1000)/1e9:6.0f} GB/s")
