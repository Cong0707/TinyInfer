import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
import tinyinfer.gemm as G
_orig = G.fp4_linear
seen = []
def w(x, w_fp4, w_scale, act_block=32):
    seen.append((tuple(x.shape), x.dtype))
    return _orig(x, w_fp4, w_scale, act_block)
G.fp4_linear = w
# moe.py 函数内 import 会拿到新值
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=list(range(40)))
with torch.inference_mode():
    for label, ids, pos in (('prefill4', list(range(100,104)), 0),):
        seen.clear()
        eng.reset()
        eng.forward(torch.tensor([ids], device='cuda'), pos)
        torch.cuda.synchronize()
        import collections
        c = collections.Counter(seen)
        print(f'RESULT {label}: calls={len(seen)} shapes={dict(list(c.items())[:4])}')
    seen.clear()
    eng.forward(torch.tensor([[500]], device='cuda'), start_pos=4)
    torch.cuda.synchronize()
    import collections
    c = collections.Counter(seen)
    print(f'RESULT decode1: calls={len(seen)} shapes={dict(list(c.items())[:4])}')