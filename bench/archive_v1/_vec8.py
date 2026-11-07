import sys
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
import torch
import tinyinfer.torch_compat  # noqa: F401
from tinyinfer import native

def bench(name, grid, block, args, n=60, warm=15):
    with torch.inference_mode():
        for _ in range(warm):
            native.launch(native.kernel("gemv_fp4_e8m0", name), grid, block, args)
        torch.cuda.synchronize()
        e0 = torch.cuda.Event(True); e1 = torch.cuda.Event(True)
        e0.record()
        for _ in range(n):
            native.launch(native.kernel("gemv_fp4_e8m0", name), grid, block, args)
        e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1)/n

N, K = 16384, 5120
Wq = torch.randint(0, 256, (N, K//2), dtype=torch.uint8, device="cuda")
sc = torch.full((N, K//32), 127, dtype=torch.uint8, device="cuda")
a = torch.randn(K, dtype=torch.float32, device="cuda")
asc = torch.ones(K//32, dtype=torch.float32, device="cuda")
y1 = torch.zeros(N, dtype=torch.bfloat16, device="cuda")
yf = torch.zeros(N, dtype=torch.bfloat16, device="cuda")
yv = torch.zeros(N, dtype=torch.bfloat16, device="cuda")
by = N*K*0.5

with torch.inference_mode():
    native.launch(native.kernel("gemv_fp4_e8m0","gemv_fp4_e8m0"),
                  ((N+15)//16,), (512,), [a,asc,Wq,sc,y1,N,K])
    native.launch(native.kernel("gemv_fp4_e8m0","gemv_fp4_fast"),
                  ((N+7)//8,), (256,), [a,asc,Wq,sc,yf,N,K])
    native.launch(native.kernel("gemv_fp4_e8m0","gemv_fp4_vec8"),
                  ((N+7)//8,), (256,), [a,asc,Wq,sc,yv,N,K])
    torch.cuda.synchronize()
print("vec8 vs table diff:", float((y1.float()-yv.float()).abs().max()))
print("vec8 vs fast  diff:", float((yf.float()-yv.float()).abs().max()))
print("  table:", y1[:3].tolist())
print("  vec8 :", yv[:3].tolist())
print()
ms1 = bench("gemv_fp4_e8m0", ((N+15)//16,), (512,), [a,asc,Wq,sc,y1,N,K])
msf = bench("gemv_fp4_fast", ((N+7)//8,), (256,), [a,asc,Wq,sc,yf,N,K])
msv = bench("gemv_fp4_vec8", ((N+7)//8,), (256,), [a,asc,Wq,sc,yv,N,K])
print(f"table: {ms1*1000:8.1f} us -> {by/(ms1/1000)/1e9:6.0f} GB/s")
print(f"fast : {msf*1000:8.1f} us -> {by/(msf/1000)/1e9:6.0f} GB/s  x{ms1/msf:.2f}")
print(f"vec8 : {msv*1000:8.1f} us -> {by/(msv/1000)/1e9:6.0f} GB/s  x{ms1/msv:.2f}")
print()
print(f"纯读取上限 (同尺寸): 212 GB/s")
