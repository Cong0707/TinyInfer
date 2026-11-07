import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=list(range(40)))
hp = eng.host_pool
print(f'RESULT pool total: {hp.total/2**30:.2f} GB, used: {hp.used/2**30:.2f} GB, free: {hp.total-hp.used}')
# 看 slices
for a in ('_slices','_spans','slices','spans'):
    v = getattr(hp, a, None)
    if v:
        print(f'RESULT {a}: {len(v)} entries')
        tot = 0
        for it in (v if isinstance(v,list) else list(v.values()))[:20]:
            try:
                sz = it[1]-it[0] if isinstance(it,tuple) else getattr(it,'nbytes',getattr(it,'size',0))
                tot += sz
            except Exception: pass
        print(f'  sum of first 20: {tot/2**30:.2f} GB')
print('RESULT attrs:', [x for x in dir(hp) if not x.startswith('__')])