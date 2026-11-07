import sys, torch, collections
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
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for i in range(5):
            eng.forward(torch.tensor([[500+i]], device='cuda'), start_pos=8+i)
        torch.cuda.synchronize()
rows = []
tot = 0.0
for ev in prof.key_averages():
    dv = getattr(ev,'self_device_time_total',0.0)/5
    if 'Memcpy' in ev.key: continue
    if dv > 100: rows.append((dv/1000, ev.key[:44], ev.count/5))
    tot += dv/1000
rows.sort(reverse=True)
print('RESULT kernels total: %.1f ms' % tot)
for dv,k,n in rows[:14]:
    print('RESULT %-46s %6.2f ms  n=%.0f' % (k, dv, n))