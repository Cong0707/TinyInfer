import sys, torch, time
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=list(range(40)))
es = eng.expert_store; ram = es.ram
cur = {'fetch':0.0}
_f = es.fetch
def wf(layer, experts, **kw):
    t0=time.perf_counter(); r=_f(layer, experts, **kw); cur['fetch']+=time.perf_counter()-t0; return r
es.fetch = wf
rows = []
with torch.inference_mode():
    eng.reset()
    eng.forward(torch.tensor([list(range(100,108))], device='cuda'), 0)
    torch.cuda.synchronize()
    for i in range(12):
        s0 = es.stats(); cur['fetch']=0.0
        torch.cuda.synchronize(); t0=time.perf_counter()
        eng.forward(torch.tensor([[500+i]], device='cuda'), start_pos=8+i)
        torch.cuda.synchronize(); tt=(time.perf_counter()-t0)*1000
        s1 = es.stats()
        db = (s1['reader']['bytes_read']-s0['reader']['bytes_read'])/1e6
        rows.append((i, tt, cur['fetch']*1000, db))
print('RESULT tok  total_ms  fetch_ms  disk_MB')
for i, tt, fw, db in rows:
    print(f'RESULT {i:3d}  {tt:7.0f}  {fw:7.0f}  {db:7.0f}')
import statistics
ts=[r[1] for r in rows]; fs=[r[2] for r in rows]; bs=[r[3] for r in rows]
print(f'RESULT mean total={statistics.mean(ts):.0f} fetch={statistics.mean(fs):.0f} ({statistics.mean(fs)/statistics.mean(ts)*100:.0f}%) disk={statistics.mean(bs):.0f}MB')