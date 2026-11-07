import sys, torch, os
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
os.environ['TI_CACHE_GB'] = '1'
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=list(range(40)))
hp = eng.host_pool
es = eng.expert_store
print(f'RESULT with TI_CACHE_GB=1: used={hp.used/2**30:.2f} GB, es._pool={type(getattr(es,"_pool",None)).__name__}')
if getattr(es,'_pool',None):
    print(f'RESULT pool capacity: {es._pool.capacity/2**30:.2f} GB')