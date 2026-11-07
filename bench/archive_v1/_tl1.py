import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
# 直接检查：不装 tilelang 能不能跑
import importlib.util
spec = importlib.util.find_spec('tilelang')
print(f'RESULT tilelang installed: {spec is not None}')
# 拦截 import 测试
import builtins
real_import = builtins.__import__
blocked = []
def guard(name, *a, **kw):
    if name.split('.')[0] == 'tilelang':
        blocked.append(name)
        raise ImportError(f'tilelang blocked: {name}')
    return real_import(name, *a, **kw)
builtins.__import__ = guard
try:
    from tinyinfer.engine import Engine
    from tinyinfer.tiers import OptLevel
    d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
    eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=list(range(40)))
    with torch.inference_mode():
        eng.reset()
        eng.forward(torch.tensor([list(range(100,108))], device='cuda'), 0)
        torch.cuda.synchronize()
    print(f'RESULT engine ran OK, tilelang imports attempted: {len(blocked)}')
    for b in sorted(set(blocked))[:8]: print(f'  blocked: {b}')
except Exception as e:
    print(f'RESULT FAILED: {repr(e)[:200]}')
    for b in sorted(set(blocked))[:8]: print(f'  blocked: {b}')