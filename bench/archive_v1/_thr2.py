import sys, torch, time
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
import os
nt = int(os.environ.get('NT','16'))
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=list(range(40)))
es = eng.expert_store
sr = es.reader
print(f'RESULT before: threads={sr.threads}')
sr.threads = nt
if sr._executor is not None:
    sr._executor.shutdown(wait=False); sr._executor = None
with torch.inference_mode():
    eng.reset()
    eng.forward(torch.tensor([list(range(100,108))], device='cuda'), 0)
    torch.cuda.synchronize()
    s1 = es.stats()
    ts=[]
    for i in range(4):
        torch.cuda.synchronize(); t0=time.perf_counter()
        eng.forward(torch.tensor([[500+i]], device='cuda'), start_pos=8+i)
        torch.cuda.synchronize(); ts.append(time.perf_counter()-t0)
    s2 = es.stats()
db = s2['reader']['bytes_read']-s1['reader']['bytes_read']
ds = s2['reader']['read_seconds']-s1['reader']['read_seconds']
print(f'RESULT threads={nt}: token={min(ts)*1000:.0f}ms  read={db/1e9:.2f}GB in {ds:.2f}s = {db/ds/1e9:.2f} GB/s')