import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=list(range(40)))
es = eng.expert_store
print('RESULT expert_slots (plan):', getattr(eng.hw_plan, 'expert_slots', None))
print('RESULT slots obj:', getattr(es, 'slots', None) is not None)
sl = getattr(es, 'slots', None)
if sl is not None:
    print('RESULT slots type:', type(sl).__name__)
    for a in ('n','size','count','capacity'):
        if hasattr(sl, a): print(f'  {a} = {getattr(sl,a)}')
    print('RESULT slots attrs:', [x for x in dir(sl) if not x.startswith('_')][:10])