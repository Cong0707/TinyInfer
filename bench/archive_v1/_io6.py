import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=list(range(40)))
hp = eng.host_pool
print(f'RESULT host_pool: {type(hp).__name__}')
for a in ('size','used','free','total','capacity','budget'):
    v = getattr(hp, a, None)
    if v is not None: print(f'  {a}: {v/2**30:.2f} GB' if isinstance(v,int) and v>1e6 else f'  {a}: {v}')
print('RESULT attrs:', [a for a in dir(hp) if not a.startswith('_')][:12])
es = eng.expert_store
print(f'RESULT es._pool: {getattr(es,"_pool",None)}')
print(f'RESULT es.capacity_entries: {es.capacity_entries}')