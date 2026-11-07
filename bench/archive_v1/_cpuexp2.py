import sys, time, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
N, K = 2304, 5120
torch.manual_seed(0)
w = torch.randint(0, 256, (N, K//2), dtype=torch.uint8)
x = torch.randn(1, K, dtype=torch.bfloat16)
LUT = torch.tensor([0.,0.5,1.,1.5,2.,3.,4.,6.,0.,-0.5,-1.,-1.5,-2.,-3.,-4.,-6.], dtype=torch.float32)
def cpu_expert(iters=5):
    t0 = time.perf_counter()
    for _ in range(iters):
        u = w.reshape(N, K//2)
        lo = LUT[(u & 0x0F).long()]
        hi = LUT[((u >> 4) & 0x0F).long()]
        wf = torch.stack([lo, hi], dim=-1).reshape(N, K)
        torch.nn.functional.linear(x.float(), wf)
    return (time.perf_counter() - t0) / iters * 1000
for n in (32, 8, 4):
    torch.set_num_threads(n)
    print(f"CPU fp4 GEMV ({n:2d} threads): {cpu_expert():.1f} ms/expert")
