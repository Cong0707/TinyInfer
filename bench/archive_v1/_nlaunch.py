import sys, torch, collections
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.kernels import native
import tinyinfer.kernels.native as NAT
cnt = collections.Counter()
_o = NAT.launch
def w(fn, grid, block, args, *a, **kw):
    cnt['total'] += 1
    return _o(fn, grid, block, args, *a, **kw)
NAT.launch = w
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=list(range(40)))
with torch.inference_mode():
    eng.reset()
    eng.forward(torch.tensor([list(range(100,108))], device='cuda'), 0)
    torch.cuda.synchronize()
    print(f'RESULT prefill native launches: {cnt["total"]}')
    cnt['total']=0
    eng.forward(torch.tensor([[500]], device='cuda'), start_pos=8)
    torch.cuda.synchronize()
    print(f'RESULT decode  native launches: {cnt["total"]}')