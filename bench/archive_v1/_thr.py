import sys, torch, time, os
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
import tinyinfer.loading as L
print('RESULT default threads in ShardReader:', L.ShardReader.__init__.__defaults__)
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=list(range(40)))
es = eng.expert_store
sr = es.reader if hasattr(es,'reader') else getattr(es, '_reader', None)
print('RESULT reader threads:', getattr(sr, 'threads', '?'))
def one_step():
    with torch.inference_mode():
        eng.reset()
        eng.forward(torch.tensor([list(range(100,108))], device='cuda'), 0)
        torch.cuda.synchronize()
        s1 = es.stats()
        ts=[]
        for i in range(3):
            torch.cuda.synchronize(); t0=time.perf_counter()
            eng.forward(torch.tensor([[500+i]], device='cuda'), start_pos=8+i)
            torch.cuda.synchronize(); ts.append(time.perf_counter()-t0)
        s2 = es.stats()
    return min(ts)*1000, s2['reader']['gbps']
for nt in (6, 10, 16):
    if sr is not None: sr.threads = nt
    for old, new in ((sr._executor if hasattr(sr,'_executor') else None), None):
        pass
    if hasattr(sr, '_executor') and sr._executor is not None:
        sr._executor.shutdown(wait=False); sr._executor = None
    ms, gbps = one_step()
    print(f'RESULT threads={nt:2d}: {ms:.0f} ms/token, reader {gbps:.2f} GB/s')