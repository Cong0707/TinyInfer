import sys, builtins, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
real = builtins.__import__
def guard(name, *a, **kw):
    if '_stable_libtorch' in name: raise ImportError('pyd blocked')
    return real(name, *a, **kw)
builtins.__import__ = guard
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=list(range(40)))
with torch.inference_mode():
    eng.reset()
    eng.forward(torch.tensor([list(range(100,108))], device='cuda'), 0)
    torch.cuda.synchronize()
    out = list(eng.stream_generate(list(range(100,108)), max_new_tokens=3, temperature=0.0))
    torch.cuda.synchronize()
print(f'RESULT no-pyd out={out}')