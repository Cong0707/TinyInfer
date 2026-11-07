import sys, torch, traceback, collections
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
import tinyinfer.budget as B
calls = []
_o = B.HostPool.alloc
def w(self, n, align=4096):
    r = _o(self, n, align)
    st = traceback.extract_stack()
    # 找第一个在 tinyinfer 里、且不是 budget.py 的帧
    for f in reversed(st[:-1]):
        if 'tinyinfer' in f.filename.replace(chr(92),'/') and 'budget.py' not in f.filename:
            calls.append((n, f.filename.split(chr(92))[-1], f.lineno, f.line or ''))
            break
    return r
B.HostPool.alloc = w
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=list(range(40)))
c = collections.defaultdict(lambda: [0,0])
for n, f, ln, line in calls:
    if n <= 100*2**20:
        k = (f, ln, line.strip()[:50])
        c[k][0] += 1; c[k][1] += n
top = sorted(c.items(), key=lambda kv: -kv[1][1])[:4]
for (f, ln, line), (cnt, tot) in top:
    print(f'RESULT {f}:{ln} {cnt}x {tot/2**30:.2f}GB | {line}')