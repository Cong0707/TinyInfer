import sys, torch, time
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=list(range(40)))
T = {}
def tk(k, dt): T[k] = T.get(k,0.0) + dt
import types
for lid, b in getattr(eng, '_blocks', {}).items():
    for name in ('hc_pre', 'hc_post', 'hc_mixes'):
        m = getattr(b, name, None)
        if not callable(m): continue
        def mk(nm, f):
            def w(*a, **kw):
                t0=time.perf_counter(); r=f(*a,**kw); tk(nm, time.perf_counter()-t0); return r
            return w
        setattr(b, name, mk(name, m))
with torch.inference_mode():
    eng.reset()
    eng.forward(torch.tensor([list(range(100,108))], device='cuda'), 0)
    torch.cuda.synchronize()
    T.clear()
    torch.cuda.synchronize(); t0=time.perf_counter()
    eng.forward(torch.tensor([[500]], device='cuda'), start_pos=8)
    torch.cuda.synchronize(); total=(time.perf_counter()-t0)*1000
print(f'RESULT total={total:.0f}ms')
for k,v in sorted(T.items(), key=lambda kv:-kv[1]):
    print(f'RESULT   {k:10s} {v*1000:6.1f}ms {v*1000/total*100:4.0f}%')