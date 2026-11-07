import sys, torch, collections
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
import tinyinfer.budget as B
import traceback
calls = []
_o = B.HostPool.alloc
def w(self, n, align=4096):
    r = _o(self, n, align)
    st = traceback.extract_stack()
    who = st[-4].name + '<-' + st[-3].name
    calls.append((n, who))
    return r
B.HostPool.alloc = w
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=list(range(40)))
small = [(n, who) for n, who in calls if n <= 100*2**20]
c = collections.Counter(who for _, who in small)
print(f'RESULT small allocs: {len(small)}, total {sum(n for n,_ in small)/2**30:.2f} GB')
for who, cnt in c.most_common(6):
    tot = sum(n for n, w in small if w == who)
    print(f'RESULT {who:32s} {cnt:5d} 次  {tot/2**30:6.2f} GB')