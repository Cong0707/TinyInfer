import sys
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
import torch
import tinyinfer.torch_compat  # noqa: F401
from tinyinfer import native

N, K = 4096, 4096
Wq = torch.randint(0, 256, (N, K//2), dtype=torch.uint8, device="cuda")
sink = torch.zeros(N, dtype=torch.uint32, device="cuda")
by = N * K * 0.5

def bench(name, grid, block, args, n=100):
    with torch.inference_mode():
        for _ in range(20):
            native.launch(native.kernel("gemv_fp4_e8m0", name), grid, block, args)
        torch.cuda.synchronize()
        e0 = torch.cuda.Event(True); e1 = torch.cuda.Event(True)
        e0.record()
        for _ in range(n):
            native.launch(native.kernel("gemv_fp4_e8m0", name), grid, block, args)
        e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1)/n

ms = bench("fp4_readonly", ((N+7)//8,), (256,), [Wq, sink, N, K])
print(f"pure fp4 read: {ms*1000:7.1f} us -> {by/(ms/1000)/1e9:6.0f} GB/s")

# same bytes as bf16 for comparison
W16 = torch.randint(0, 65535, (N, K), dtype=torch.uint16, device="cuda").view(torch.bfloat16)
sink2 = torch.zeros(N, dtype=torch.float32, device="cuda")
x = torch.randn(K, dtype=torch.bfloat16, device="cuda")
ms2 = bench("fp4_readonly", ((N+7)//8,), (256,), [Wq, sink, N, K])
print(f"(data size {by/2**20:.1f} MB, card L2 is ~32 MB)")
