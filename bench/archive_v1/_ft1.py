import sys, torch, time
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
p = tok('Hello', return_tensors='pt').input_ids[0].tolist()
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=list(range(40)))
# 给 ExpertStore.fetch 打点
import tinyinfer.loading as L
es = eng.expert_store
timings = {'fetch':0.0,'n':0}
orig_fetch = es.fetch
def wrapped(*a, **kw):
    t0 = time.perf_counter()
    r = orig_fetch(*a, **kw)
    timings['fetch'] += time.perf_counter()-t0
    timings['n'] += 1
    return r
es.fetch = wrapped
# prefetch 也打点
if hasattr(es, 'prefetch'):
    orig_pf = es.prefetch
    timings['pf'] = 0.0
    def wpf(*a, **kw):
        t0 = time.perf_counter()
        r = orig_pf(*a, **kw)
        timings['pf'] += time.perf_counter()-t0
        return r
    es.prefetch = wpf
with torch.inference_mode():
    eng.reset(); list(eng.stream_generate(p, max_new_tokens=2, temperature=0.0))
    for k in timings: timings[k] = 0 if k!='pf' else 0.0
    t0 = time.perf_counter()
    list(eng.stream_generate(p, max_new_tokens=1, temperature=0.0))
    tt = time.perf_counter()-t0
print(f'RESULT step={tt*1000:.0f}ms fetch={timings["fetch"]*1000:.0f}ms n={timings["n"]} prefetch={timings.get("pf",0)*1000:.0f}ms')
print(f'RESULT rest={tt*1000 - timings["fetch"]*1000 - timings.get("pf",0)*1000:.0f}ms')