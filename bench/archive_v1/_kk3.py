import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=list(range(40)))
with torch.inference_mode():
    eng.reset()
    for _ in range(2): eng.forward(torch.tensor([[500]], device='cuda'), start_pos=8)
    torch.cuda.synchronize()
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
        eng.forward(torch.tensor([[501]], device='cuda'), start_pos=9)
        torch.cuda.synchronize()
found = False
for ev in prof.events():
    k = getattr(ev,'key','')
    if 'kernel_kernel' in str(k):
        found = True
        print(f'RESULT name={k}  dur={ev.device_time_total/1000:.2f}ms')
if not found:
    print('RESULT 本次运行没有 kernel_kernel')