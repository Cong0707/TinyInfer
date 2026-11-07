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
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        eng.forward(torch.tensor([[500]], device='cuda'), start_pos=8)
        torch.cuda.synchronize()
for ev in prof.key_averages():
    if 'kernel_kernel' in ev.key:
        dv = getattr(ev,'self_device_time_total',0.0)/1000
        print(f'RESULT kernel_kernel: {dv:.2f} ms, count={ev.count}')
        print(f'RESULT input shapes: {ev.input_shapes}')
# 看是哪个模块调的
    else: continue