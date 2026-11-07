import sys, torch, traceback
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
import tinyinfer.budget as B
import tinyinfer.engine as E
seq = []
_o = B.HostPool.alloc
def w(self, n, align=4096):
    r = _o(self, n, align)
    if n > 1_000_000:
        for f in reversed(traceback.extract_stack()[:-1]):
            if 'tinyinfer' in f.filename.replace(chr(92),'/') and 'budget.py' not in f.filename:
                seq.append((n, f.filename.split(chr(92))[-1]+':'+str(f.lineno)))
                break
    return r
B.HostPool.alloc = w
from tinyinfer.tiers import OptLevel
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
eng = E.Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=list(range(40)))
print(f'RESULT big allocs in order: {len(seq)}')
for n, who in seq[:12]:
    print(f'RESULT {n/2**20:8.1f} MB  {who}')