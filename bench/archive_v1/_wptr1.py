import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=[0])
b = eng.block(0)
paths = b._weight_paths()
for (owner, name, t) in paths[:6]:
    ptr = int(t.data_ptr()) if hasattr(t,'data_ptr') else -1
    print(f'RESULT {name[:36]:38s} ptr={ptr:14d} pinned={t.is_pinned()} dev={t.device}')
pool = getattr(eng, 'host_pool', None)
print('RESULT pool base:', getattr(pool, '_base', None), 'total', pool.total/2**30 if pool else 0)