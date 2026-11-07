import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=list(range(40)))
with torch.inference_mode():
    eng.reset()
    eng.forward(torch.tensor([list(range(100,108))], device='cuda'), 0)
    torch.cuda.synchronize()
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for i in range(3):
            eng.forward(torch.tensor([[500+i]], device='cuda'), start_pos=8+i)
        torch.cuda.synchronize()
    rows=[]
    for ev in prof.key_averages():
        dv = getattr(ev,'self_device_time_total',0.0)
        if dv > 500: rows.append((dv, ev.key[:50], ev.count))
    rows.sort(reverse=True)
    tot = sum(r[0] for r in rows)
    print(f'RESULT total device time across 3 steps: {tot/1000:.1f}ms')
    for dv,k,n in rows[:10]:
        print(f'RESULT {k:52s} {dv/1000:7.1f}ms  n={n}')