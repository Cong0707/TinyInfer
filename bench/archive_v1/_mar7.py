import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
import tinyinfer.deepseek.moe as MOE
shapes = []
_orig = MOE.MoE.forward
def w(self, x, *a, **kw):
    shapes.append((self.layer_id, tuple(x.shape)))
    return _orig(self, x, *a, **kw)
MOE.MoE.forward = w
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=list(range(40)))
with torch.inference_mode():
    for n in (4, 8, 16):
        shapes.clear()
        eng.reset()
        eng.forward(torch.tensor([list(range(100,100+n))], device='cuda'), 0)
        torch.cuda.synchronize()
        uniq = sorted(set(s[1][0] for s in shapes))
        print(f'RESULT prefill({n}): calls={len(shapes)} M_values={uniq}')