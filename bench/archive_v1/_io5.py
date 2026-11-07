import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=list(range(40)))
es = eng.expert_store
pool = getattr(es, '_pool', None)
print(f'RESULT pool: {type(pool).__name__ if pool else None}')
if pool:
    for a in ('free_bytes','used_bytes','total_bytes','capacity'):
        print(f'  {a}: {getattr(pool, a, "?")}')
    # 试几次 take
    try:
        s = pool.take(18_800_000)
        print(f'  take ok: {type(s).__name__}')
    except Exception as e:
        print(f'  take FAILED: {repr(e)[:90]}')
ex = getattr(es.reader, 'io_executor', None)
print(f'RESULT io_executor: {ex is not None}')
if ex: print(f'  max_workers: {ex()._max_workers}')