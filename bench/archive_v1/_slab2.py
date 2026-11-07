import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
import tinyinfer.budget as B
calls = []
_o = B.HostPool.alloc
import traceback
def w(self, n, align=4096):
    r = _o(self, n, align)
    st = traceback.extract_stack()
    who = '/'.join(f'{f.name}' for f in st[-4:-1])
    calls.append((n, r is not None, who))
    return r
B.HostPool.alloc = w
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=list(range(40)))
big = [(n, ok, who) for n, ok, who in calls if n > 100*2**20]
print(f'RESULT alloc calls: {len(calls)}, big (>100MB): {len(big)}')
for n, ok, who in big[:8]:
    print(f'RESULT {n/2**30:6.2f} GB {"ok" if ok else "FAIL"}  {who}')
tot = sum(n for n, ok, who in big if ok)
print(f'RESULT big total: {tot/2**30:.2f} GB')