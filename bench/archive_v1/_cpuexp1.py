import sys, time, torch, numpy as np
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
# 模拟一个专家：w1 [2304, 5120] fp4 + e8m0 scale，x [1, 5120] bf16
N, K = 2304, 5120
torch.manual_seed(0)
# fp4 packed: [N, K/2] uint8
w = torch.randint(0, 256, (N, K//2), dtype=torch.uint8)
x = torch.randn(1, K, dtype=torch.bfloat16)
print("weight MB:", N*K//2/1e6, " x MB:", x.numel()*2/1e6)

# CPU 端：dequant 表 + matmul（模拟最朴素的实现）
LUT = torch.tensor([0.,0.5,1.,1.5,2.,3.,4.,6.,0.,-0.5,-1.,-1.5,-2.,-3.,-4.,-6.], dtype=torch.float32)
def cpu_expert(w, x, iters=3):
    t0 = time.perf_counter()
    for _ in range(iters):
        u = w.reshape(-1, 2)
        lo = LUT[(u[:, 0] & 0x0F)]
        hi = LUT[(u[:, 0] >> 4)]
        # interleave back to K
        wf = torch.stack([lo.view(N, K//2), hi.view(N, K//2)], dim=-1).reshape(N, K)
        y = torch.nn.functional.linear(x.float(), wf)
    return (time.perf_counter() - t0) / iters * 1000

torch.set_num_threads(32)
print(f"CPU fp4 GEMV (32 threads): {cpu_expert(w, x):.1f} ms/expert")
torch.set_num_threads(8)
print(f"CPU fp4 GEMV ( 8 threads): {cpu_expert(w, x):.1f} ms/expert")
