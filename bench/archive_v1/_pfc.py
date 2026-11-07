import sys, torch, time
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
import os
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=list(range(40)))
ids = list(range(100, 108))
with torch.inference_mode():
    eng.reset(); eng.forward(torch.tensor([ids], device='cuda'), 0); torch.cuda.synchronize()
    ts=[]
    for _ in range(3):
        eng.reset(); torch.cuda.synchronize(); t0=time.perf_counter()
        eng.forward(torch.tensor([ids], device='cuda'), 0)
        torch.cuda.synchronize(); ts.append(time.perf_counter()-t0)
print(f'RESULT prefill(8) cache={os.environ.get("TI_MARLIN_CACHE","64")}: {min(ts)*1000:.0f} ms')