import sys, torch, time
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.gemm import fp8_linear

def bench(f, R=50, w_=10):
    for _ in range(w_): f()
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(R): f()
    torch.cuda.synchronize(); return (time.perf_counter()-t0)/R*1e6

# 真实形状
for (N,K,name) in ((32768,1280,"wq_b"), (5120,2304,"shared.w2"), (2304,5120,"shared.w1")):
    w = torch.randn(N,K,dtype=torch.float16,device="cuda").to(torch.float8_e4m3fn)
    s = torch.ones(N//32,K//32,dtype=torch.float8_e8m0fnu,device="cuda")
    x = torch.randn(1,K,dtype=torch.bfloat16,device="cuda")
    t = bench(lambda: fp8_linear(x,w,s))
    mb = N*K/1e6
    print(f"{name:12s} [{N:5d},{K:4d}] {t:8.0f} us  {mb/t*1e3:5.0f} GB/s")
