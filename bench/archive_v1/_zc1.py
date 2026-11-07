import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=list(range(40)))
hp = eng.host_pool
print(f'RESULT pool._base = {getattr(hp,"_base",None)}')
b = eng.block(10)
paths = b._weight_paths()
print(f'RESULT layer10 tensors: {len(paths)}')
import ctypes
for owner, name, t in paths[:4]:
    dp = int(t.data_ptr())
    base = getattr(hp, '_base', 0)
    print(f'RESULT {name:22s} dptr={dp} diff={dp-base if base else "?"}')