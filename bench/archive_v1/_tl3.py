import sys, torch, builtins
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
real = builtins.__import__
def guard(name, *a, **kw):
    if name.split('.')[0]=='tilelang': raise ImportError('tilelang removed from this tree')
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
    ids = list(range(100,108))
    t0=__import__('time').perf_counter()
    out = list(eng.stream_generate(ids, max_new_tokens=3, temperature=0.0))
    torch.cuda.synchronize()
print(f'RESULT OK: out={out} {( __import__("time").perf_counter()-t0)*1000:.0f}ms')