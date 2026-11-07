import sys, torch, collections
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
p = tok('Hello', return_tensors='pt').input_ids[0].tolist()
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=list(range(40)))
# 统计 _flat_apply 的调用和字节
import tinyinfer.engine as E
stats = collections.Counter()
of = eng._flat_apply
def wf(layer, buf_idx):
    lay, bufs, host = eng._flat_state
    offs, total = lay[layer]
    stats['flat_layers'] += 1
    stats['flat_bytes'] += total
    return of(layer, buf_idx)
eng._flat_apply = wf
# 统计专家传输
es = eng.expert_store
ofetch = es.fetch
def wfetch(*a, **kw):
    r = ofetch(*a, **kw)
    try:
        n = sum(1 for _ in r) if hasattr(r,'__iter__') else 1
    except Exception:
        n = 1
    stats['fetch_calls'] += 1
    return r
es.fetch = wfetch
with torch.inference_mode():
    eng.reset(); list(eng.stream_generate(p, max_new_tokens=2, temperature=0.0))
    for k in list(stats): stats[k] = 0
    t0 = __import__('time').perf_counter()
    list(eng.stream_generate(p, max_new_tokens=1, temperature=0.0))
    torch.cuda.synchronize()
    dt = __import__('time').perf_counter()-t0
print(f'RESULT step={dt*1000:.0f}ms')
print(f'RESULT flat_apply calls={stats["flat_layers"]} bytes={stats["flat_bytes"]/2**30:.2f}GB')
print(f'RESULT fetch calls={stats["fetch_calls"]}')
s = es.stats()
print(f'RESULT ram hits={s["ram"]["hits"]} misses={s["ram"]["misses"]}')
print(f'RESULT 估算专家字节 {(s["ram"]["hits"]+s["ram"]["misses"])*18.8/1024:.2f}GB')