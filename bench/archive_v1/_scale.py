import sys
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
import torch
import tinyinfer.torch_compat  # noqa: F401
from tinyinfer import native
MB = 2**20

def bench(name, grid, block, args, n=40, warm=10):
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

print("=== pure fp4 read at increasing sizes (L2 is 32 MB) ===")
for N, K in ((4096, 4096), (16384, 5120), (65536, 5120), (131072, 5120)):
    Wq = torch.randint(0, 256, (N, K//2), dtype=torch.uint8, device="cuda")
    sink = torch.zeros(N, dtype=torch.uint32, device="cuda")
    by = N * K * 0.5
    ms = bench("fp4_readonly", ((N+7)//8,), (256,), [Wq, sink, N, K])
    gbs = by/(ms/1000)/1e9
    tag = "L2" if by < 32*MB else "DRAM"
    print(f"  {N:6d}x{K}: {by/MB:7.1f} MB ({tag})  {ms*1000:8.1f} us -> {gbs:5.0f} GB/s")
    del Wq, sink
    torch.cuda.empty_cache()

print()
print("=== bf16 GEMV at same byte sizes for comparison ===")
for N, K in ((4096, 4096), (16384, 5120), (65536, 5120)):
    x = torch.randn(K, dtype=torch.bfloat16, device="cuda")
    W = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
    y = torch.zeros(N, dtype=torch.float32, device="cuda")
    by = N * K * 2
    ms = bench("fp4_readonly", ((N+7)//8,), (256,), [Wq if False else y, y, N, K]) if False else None
    # use the real bf16 kernel
    with torch.inference_mode():
        for _ in range(10):
            native.launch(native.kernel("gemv_bf16","gemv_bf16"), ((N+7)//8,),(256,),[x,W,y,N,K])
        torch.cuda.synchronize()
        e0 = torch.cuda.Event(True); e1 = torch.cuda.Event(True)
        e0.record()
        for _ in range(40):
            native.launch(native.kernel("gemv_bf16","gemv_bf16"), ((N+7)//8,),(256,),[x,W,y,N,K])
        e1.record(); torch.cuda.synchronize()
    ms = e0.elapsed_time(e1)/40
    gbs = by/(ms/1000)/1e9
    tag = "L2" if by < 32*MB else "DRAM"
    print(f"  {N:6d}x{K}: {by/MB:7.1f} MB ({tag})  {ms*1000:8.1f} us -> {gbs:5.0f} GB/s")
    del x, W, y
    torch.cuda.empty_cache()
