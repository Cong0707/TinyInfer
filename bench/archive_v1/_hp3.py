import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=list(range(40)))
es = eng.expert_store
ram = es.ram
print(f'RESULT ram._pool: {type(getattr(ram,"_pool",None)).__name__}')
p = getattr(ram, '_pool', None)
if p:
    print(f'RESULT pool capacity: {getattr(p,"capacity",0)/2**30:.2f} GB')
    print(f'RESULT pool slots: {getattr(p,"n_slots",getattr(p,"slots","?"))}')
print(f'RESULT ram capacity_entries: {ram.capacity_entries}')
print(f'RESULT es._read_many_pooled exists: {hasattr(es,"_read_many_pooled")}')