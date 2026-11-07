import sys, torch, time
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=list(range(40)))
es = eng.expert_store
sr = es.reader
print(f'RESULT reader threads={sr.threads} executor={type(sr._executor).__name__ if sr._executor else None}')
# 看 executor 的 max_workers
if sr._executor: print(f'RESULT max_workers={sr._executor._max_workers}')
# 看 read_many_pinned 的 workers 计算
import inspect
src = inspect.getsource(es.read_many_pinned) if hasattr(es,'read_many_pinned') else ''
print('RESULT read_many_pinned:', src[:300] if src else 'n/a')