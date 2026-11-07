import sys, torch, time
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=list(range(40)))
def bench(n, label, reps=3):
    ids = list(range(100, 100+n))
    with torch.inference_mode():
        eng.reset()
        eng.forward(torch.tensor([ids], device='cuda'), 0)   # warm
        torch.cuda.synchronize()
        ts = []
        for _ in range(reps):
            eng.reset()
            torch.cuda.synchronize(); t0=time.perf_counter()
            eng.forward(torch.tensor([ids], device='cuda'), 0)
            torch.cuda.synchronize(); ts.append(time.perf_counter()-t0)
    print(f'RESULT {label} prefill({n}) = {min(ts)*1000:.0f}ms')
bench(4, 'base')
bench(8, 'base')
# decode 也测
with torch.inference_mode():
    eng.reset()
    eng.forward(torch.tensor([list(range(100,108))], device='cuda'), 0)
    torch.cuda.synchronize()
    ts=[]
    for i in range(4):
        torch.cuda.synchronize(); t0=time.perf_counter()
        eng.forward(torch.tensor([[500+i]], device='cuda'), start_pos=8+i)
        torch.cuda.synchronize(); ts.append(time.perf_counter()-t0)
    print(f'RESULT base decode = {min(ts[1:])*1000:.0f}ms')