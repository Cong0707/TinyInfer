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
ya = torch.zeros(N, dtype=torch.bfloat16, device="cuda")
by = N*K*0.5

with torch.inference_mode():
    native.launch(native.kernel("gemv_fp4_e8m0","gemv_fp4_e8m0"),
                  ((N+15)//16,), (512,), [a,asc,Wq,sc,y1,N,K])
    native.launch(native.kernel("gemv_fp4_e8m0","gemv_fp4_e2m1"),
                  ((N+7)//8,), (256,), [a,asc,Wq,sc,ya,N,K])
    torch.cuda.synchronize()
print("table vs ALU diff amax:", float((y1.float()-ya.float()).abs().max()))
print("  v1[0:3]:", y1[:3].tolist())
print("  ALU[0:3]:", ya[:3].tolist())

ms1 = bench("gemv_fp4_e8m0", ((N+15)//16,), (512,), [a,asc,Wq,sc,y1,N,K])
msa = bench("gemv_fp4_e2m1", ((N+7)//8,), (256,), [a,asc,Wq,sc,ya,N,K])
print(f"table: {ms1*1000:8.1f} us -> {by/(ms1/1000)/1e9:6.0f} GB/s")
print(f"ALU  : {msa*1000:8.1f} us -> {by/(msa/1000)/1e9:6.0f} GB/s  x{ms1/msa:.2f}")
