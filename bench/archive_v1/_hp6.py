import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=list(range(40)))
hp = eng.host_pool
es = eng.expert_store
print(f'RESULT pool: total={hp.total/2**30:.2f} used={hp.used/2**30:.2f} free={hp.free/2**30:.2f} GB')
print(f'RESULT cache_bytes (plan): {es.ram.capacity_entries} entries')
print(f'RESULT cache pool: {getattr(es.ram, "_pool", None) is not None}')
cp = getattr(es.ram, '_pool', None)
if cp: print(f'RESULT cache pool capacity: {getattr(cp,"capacity",0)/2**30:.2f} GB')
# 估算：cache 占了 16GB，slab 需要 5.34GB，合计 21.3 > 20.47