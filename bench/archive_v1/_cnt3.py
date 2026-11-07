import sys, torch, collections
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
import tinyinfer.kernels.native as NAT
cnt = collections.Counter(); kn = {}
_k = NAT.kernel
def wk(mod, sym):
    f = _k(mod, sym); kn[id(f)] = f'{mod}.{sym}'; return f
NAT.kernel = wk
_o = NAT.launch
def w2(fn, grid, block, args, *a, **kw):
    cnt[kn.get(id(fn), '?')] += 1
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
for k,v in cnt.most_common(10): print(f'RESULT {k:36s} {v:5d}')
print('RESULT total', sum(cnt.values()))