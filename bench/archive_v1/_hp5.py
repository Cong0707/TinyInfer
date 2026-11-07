import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=list(range(40)))
hp = getattr(eng, 'host_pool', 'MISSING')
print(f'RESULT host_pool: {type(hp).__name__ if hp is not None else None}')
if hp is None or hp == 'MISSING':
    print('RESULT => pool 没建成，slab 构建被整个跳过')
else:
    print(f'RESULT total={hp.total/2**30:.2f}GB used={hp.used/2**30:.2f}GB free={hp.free/2**30:.2f}GB')
from tinyinfer.budget import budget as hb
b = hb()
print(f'RESULT budget total={b.total/2**30:.1f}GB room={b.room()/2**30:.1f}GB')