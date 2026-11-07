import sys, torch, collections
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
import tinyinfer.kernels.native as NAT
cnt = collections.Counter()
_o = NAT.launch
def w(fn, grid, block, args, *a, **kw):
    try: name = getattr(fn, '__name__', None) or (fn[0] if isinstance(fn,(tuple,list)) else str(fn))
    except Exception: name = '?'
    cnt[str(name)[:40]] += 1
    return _o(fn, grid, block, args, *a, **kw)
NAT.launch = w
_k = NAT.kernel
kn = {}
def wk(mod, sym):
    f = _k(mod, sym)
    kn[id(f)] = f'{mod}.{sym}'
    return f
NAT.kernel = wk
def w2(fn, grid, block, args, *a, **kw):
    cnt[kn.get(id(fn), str(fn)[:30])] += 1
    return _o(fn, grid, block, args, *a, **kw)
NAT.launch = w2
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=list(range(40)))
with torch.inference_mode():
    eng.reset()
    eng.forward(torch.tensor([list(range(100,108))], device='cuda'), 0)
    torch.cuda.synchronize()
    cnt.clear()
    eng.forward(torch.tensor([[500]], device='cuda'), start_pos=8)
    torch.cuda.synchronize()
for k,v in cnt.most_common(12):
    print(f'RESULT {k:34s} {v:5d}')
print('RESULT total', sum(cnt.values()))