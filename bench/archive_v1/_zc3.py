import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=list(range(40)))
b = eng.block(10)
paths = b._weight_paths()
ptrs = [int(t.data_ptr()) for _,_,t in paths]
base = min(ptrs)
offs = [p - base for p in ptrs]
sizes = [t.numel()*t.element_size() for _,_,t in paths]
ends = [o+n for o,n in zip(offs,sizes)]
span = max(ends)
tot = sum(sizes)
print(f'RESULT base=min(data_ptr)')
print(f'  span = {span/2**20:.1f} MB')
print(f'  data = {tot/2**20:.1f} MB')
print(f'  slack= {(span-tot)/2**20:.1f} MB')
pos = sorted(zip(offs, ends))
overlap = any(pos[k][1] > pos[k+1][0] for k in range(len(pos)-1))
print(f'  overlap = {overlap}')
print(f'  tensors = {len(paths)}')