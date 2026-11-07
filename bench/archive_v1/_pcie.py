import torch, time
torch.cuda.init()
GB = 1<<30
# 模拟 streamed layer 的 H2D：pinned 源 -> 显存
for mb in (166, 463):
    n = mb*1024*1024//2
    src_pin = torch.empty(n, dtype=torch.bfloat16, pin_memory=True)
    src_page = torch.empty(n, dtype=torch.bfloat16)
    dst = torch.empty(n, dtype=torch.bfloat16, device="cuda")
    for name, src in (("pinned", src_pin), ("pageable", src_page)):
        for _ in range(3): dst.copy_(src, non_blocking=(name=="pinned"))
        torch.cuda.synchronize()
        t0=time.perf_counter()
        for _ in range(10): dst.copy_(src, non_blocking=(name=="pinned"))
        torch.cuda.synchronize()
        dt=(time.perf_counter()-t0)/10
        print(f"{mb:4d}MB {name:9s}: {dt*1000:7.2f} ms  {mb/1024/dt:6.1f} GB/s")
