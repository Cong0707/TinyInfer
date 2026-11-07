import sys, torch, traceback, collections
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
import tinyinfer.budget as B
calls = []
_o = B.HostPool.alloc
def w(self, n, align=4096):
    r = _o(self, n, align)
    st = traceback.extract_stack()
    calls.append((n, st[-4].lineno, st[-4].name, st[-3].filename.split(chr(92))[-1]))
    return r
B.HostPool.alloc = w
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=list(range(40)))
c = collections.Counter((f, ln) for n, ln, nm, f in calls if n <= 100*2**20 and nm == '__init__')
print(f'RESULT distinct (file, line): {len(c)}')
for (f, ln), cnt in c.most_common(5):
    tot = sum(n for n, l2, nm, f2 in calls if l2==ln and f2==f and nm=='__init__')
    print(f'RESULT {f}:{ln}  {cnt} 次, {tot/2**30:.2f} GB')