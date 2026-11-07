import sys, torch, builtins, traceback
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
real_import = builtins.__import__
sites = []
def guard(name, *a, **kw):
    if name.split('.')[0] == 'tilelang':
        st = traceback.extract_stack()
        for f in reversed(st[:-1]):
            if 'tinyinfer' in f.filename.replace(chr(92),'/') or f.filename.split(chr(92))[-1] in ('gemm.py','quant.py','sparse_attn.py'):
                sites.append(f.filename.split(chr(92))[-1]+':'+str(f.lineno)+' '+str(f.line or '')[:60])
                break
        raise ImportError('blocked')
    return real_import(name, *a, **kw)
builtins.__import__ = guard
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=list(range(40)))
with torch.inference_mode():
    eng.reset()
    eng.forward(torch.tensor([list(range(100,108))], device='cuda'), 0)
    torch.cuda.synchronize()
for s in sorted(set(sites)): print('RESULT', s)