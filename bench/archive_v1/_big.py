import sys
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
import torch
import tinyinfer.torch_compat  # noqa: F401
from tinyinfer import native

def bench(grid, block, args, shared=0, n=80):
    with torch.inference_mode():
        for _ in range(15):
            native.launch(native.kernel("gemv_fp4_e8m0","gemv_fp4_e8m0_v3"), grid, block, args, shared=shared)
        torch.cuda.synchronize()
        e0 = torch.cuda.Event(True); e1 = torch.cuda.Event(True)
        e0.record()
        for _ in range(n):
            native.launch(native.kernel("gemv_fp4_e8m0","gemv_fp4_e8m0_v3"), grid, block, args, shared=shared)
        e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1)/n

print("testing at real expert sizes (bigger -> real DRAM traffic)")
print()
for N, K in ((2304*2, 5120), (7168, 5120), (16384, 5120)):
    Wq = torch.randint(0, 256, (N, K//2), dtype=torch.uint8, device="cuda")
    sc = torch.full((N, K//32), 127, dtype=torch.uint8, device="cuda")
    a = torch.randn(K, dtype=torch.float32, device="cuda")
    asc = torch.ones(K//32, dtype=torch.float32, device="cuda")
    y = torch.zeros(N, dtype=torch.bfloat16, device="cuda")
    by = N*K*0.5
    ms = bench(((N+7)//8,), (256,), [a,asc,Wq,sc,y,N,K], shared=K*4)
    print(f"  {N}x{K}: {by/2**20:6.1f} MB, {ms*1000:7.1f} us -> {by/(ms/1000)/1e9:5.0f} GB/s")
    del Wq, sc, a, asc, y
    torch.cuda.empty_cache()

# compare with a plain bf16 gemv at the same size for reference
print()
print("reference: bf16 gemv at comparable byte size")
N, K = 4096, 4096
x = torch.randn(K, dtype=torch.bfloat16, device="cuda")
W = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
y2 = torch.zeros(N, dtype=torch.float32, device="cuda")
ms = bench(((N+7)//8,), (256,), [x, W, y2, N, K])  # wrong kernel for this, skip
