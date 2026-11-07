import sys, torch, os, time
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=list(range(40)))
es = eng.expert_store
stats = {'n':0,'miss':0,'hit':0,'t':0.0}
orig = es._fetch_inner
def w(layer, experts, **kw):
    ex = [int(e) for e in experts]
    _have = es.ram.contains_many(layer, ex)
    m = sum(1 for e in ex if not _have.get(e))
    t0 = time.perf_counter()
    r = orig(layer, experts, **kw)
    stats['t'] += time.perf_counter()-t0
    stats['n'] += 1; stats['miss'] += m; stats['hit'] += len(ex)-m
    return r
es._fetch_inner = w
for n in (1, 8):
    ids = list(range(100, 100+n))
    for k in stats: stats[k] = 0 if k=='t' else 0
    with torch.inference_mode():
        eng.reset()
        list(eng.stream_generate(ids, max_new_tokens=2, temperature=0.0))
        for k in stats: stats[k] = 0 if k=='t' else 0
        list(eng.stream_generate(ids, max_new_tokens=1, temperature=0.0))
    print(f'RESULT len={n}: calls={stats["n"]} hit={stats["hit"]} miss={stats["miss"]} t={stats["t"]*1000:.0f}ms')