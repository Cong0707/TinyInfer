import torch, time
torch.cuda.init()
for gb in (16, 20, 24, 28, 30, 32, 34):
    try:
        n = gb * (1<<30)
        t0 = time.perf_counter()
        b = torch.empty(n, dtype=torch.uint8, pin_memory=True)
        dt = time.perf_counter() - t0
        print(f"{gb:2d}GB pinned: OK  ({dt*1000:.0f} ms)")
        del b
    except Exception as e:
        print(f"{gb:2d}GB pinned: FAIL {type(e).__name__}: {str(e)[:80]}")
