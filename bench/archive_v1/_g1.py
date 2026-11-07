import sys, torch, time, collections
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=list(range(40)))
# 用 CUDA event 测纯 GPU 时间，避开 CPU 侧干扰
with torch.inference_mode():
    eng.reset()
    eng.forward(torch.tensor([list(range(100,108))], device='cuda'), 0)
    torch.cuda.synchronize()
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for i in range(3):
            eng.forward(torch.tensor([[500+i]], device='cuda'), start_pos=8+i)
        torch.cuda.synchronize()
rows = []
for ev in prof.key_averages():
    dv = getattr(ev,'self_device_time_total',0.0)
    if dv > 300: rows.append((dv/3, ev.key[:46], ev.count/3))
rows.sort(reverse=True)
tot = sum(r[0] for r in rows)
print(f'RESULT GPU total/step: {tot/1000:.1f} ms')
for dv,k,n in rows[:12]:
    print(f'RESULT {k:48s} {dv/1000:7.1f} ms  n={n:.0f}')