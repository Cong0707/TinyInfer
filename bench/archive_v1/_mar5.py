import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
import tinyinfer.deepseek.gemv_fp4_marlin as MAR
calls = {'marlin': 0, 'fp4_linear': 0}
_orig_marlin = MAR.gemv_fp4_fast
def wm(*a, **kw):
    calls['marlin'] += 1
    return _orig_marlin(*a, **kw)
MAR.gemv_fp4_fast = wm
import tinyinfer.gemm as G
_orig_lin = G.fp4_linear
def wl(*a, **kw):
    calls['fp4_linear'] += 1
    return _orig_lin(*a, **kw)
G.fp4_linear = wl
import tinyinfer.deepseek.moe as MOE
MOE.fp4_linear = wl   # moe.py 是 `from ..gemm import fp4_linear`，模块级引用
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=list(range(40)))
ids = list(range(100, 104))
with torch.inference_mode():
    eng.reset()
    eng.forward(torch.tensor([ids], device='cuda'), 0)
    torch.cuda.synchronize()
    print(f'RESULT prefill(4): fp4_linear={calls["fp4_linear"]} marlin={calls["marlin"]}')
    calls['fp4_linear']=0; calls['marlin']=0
    eng.forward(torch.tensor([[500]], device='cuda'), start_pos=4)
    torch.cuda.synchronize()
    print(f'RESULT decode(1): fp4_linear={calls["fp4_linear"]} marlin={calls["marlin"]}')