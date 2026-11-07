import sys, torch, time
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
memcpy = 0.0; kernels = 0.0
for ev in prof.key_averages():
    dv = getattr(ev,'self_device_time_total',0.0)/5
    if 'Memcpy' in ev.key or 'memcpy' in ev.key: memcpy += dv
    else: kernels += dv
print('RESULT per-step: memcpy=%.1f ms  kernels=%.1f ms  total=%.1f ms' % (memcpy/1000, kernels/1000, (memcpy+kernels)/1000))
print('RESULT 若零 memcpy: %.1f ms = %.1f tok/s' % (kernels/1000, 1000/(kernels/1000)))