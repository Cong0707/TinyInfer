import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=list(range(40)))
with torch.inference_mode():
    eng.reset()
    for _ in range(3): eng.forward(torch.tensor([[500]], device='cuda'), start_pos=8)
    torch.cuda.synchronize()
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for i in range(3): eng.forward(torch.tensor([[500+i]], device='cuda'), start_pos=9+i)
        torch.cuda.synchronize()
rows=[]
for ev in prof.key_averages():
    d_ = getattr(ev,'self_device_time_total',0)/3/1000
    if d_ > 0.3: rows.append((d_, ev.key, ev.count/3))
rows.sort(reverse=True)
print('RESULT top kernels per step:')
for d_, k, c in rows[:10]:
    print('RESULT %8.2f ms  n=%-4.0f %s' % (d_, c, k[:60]))