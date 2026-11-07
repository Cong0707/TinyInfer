import sys, torch, time
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=list(range(40)))
def bench(n, label, reps=3):
    ids = list(range(100, 100+n))
    with torch.inference_mode():
        eng.reset(); eng.forward(torch.tensor([ids], device='cuda'), 0); torch.cuda.synchronize()
        ts=[]
        for _ in range(reps):
            eng.reset(); torch.cuda.synchronize(); t0=time.perf_counter()
            eng.forward(torch.tensor([ids], device='cuda'), 0)
            torch.cuda.synchronize(); ts.append(time.perf_counter()-t0)
    return min(ts)*1000
p4 = bench(4,'x'); p8 = bench(8,'x')
with torch.inference_mode():
    eng.reset(); eng.forward(torch.tensor([list(range(100,108))], device='cuda'), 0); torch.cuda.synchronize()
    ts=[]
    for i in range(4):
        torch.cuda.synchronize(); t0=time.perf_counter()
        eng.forward(torch.tensor([[500+i]], device='cuda'), start_pos=8+i)
        torch.cuda.synchronize(); ts.append(time.perf_counter()-t0)
print(f'RESULT new prefill(4)={p4:.0f}ms prefill(8)={p8:.0f}ms decode={min(ts[1:])*1000:.0f}ms')
print(f'RESULT delta: p4 {p4-1513:+.0f}ms  p8 {p8-2412:+.0f}ms  decode {min(ts[1:])*1000-758:+.0f}ms')