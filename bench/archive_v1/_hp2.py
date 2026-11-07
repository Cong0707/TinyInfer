import sys, os
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
import tinyinfer.engine as E
import tinyinfer.budget as B
_orig_alloc = B.HostPool.alloc
calls = []
def wa(self, n, align=4096):
    r = _orig_alloc(self, n, align)
    calls.append((n, r is not None, self.free))
    return r
B.HostPool.alloc = wa
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=list(range(40)))
print(f'RESULT alloc calls: {len(calls)}')
for n, ok, fr in calls[:8]:
    print(f'  {n/2**30:6.2f} GB -> {"ok" if ok else "FAIL"}  free_before={fr/2**30:.2f} GB')