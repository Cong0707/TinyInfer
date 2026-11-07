import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
import tinyinfer.deepseek.gemv_fp4_marlin as MAR
calls = {'n':0, 'repack':0}
_o = MAR.gemv_fp4_fast
_op = MAR._prepared
def wp(*a, **kw):
    calls['repack'] += 1
    return _op(*a, **kw)
MAR._prepared = wp
def w(*a, **kw):
    calls['n'] += 1
    return _o(*a, **kw)
MAR.gemv_fp4_fast = w
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=list(range(40)))
with torch.inference_mode():
    eng.reset()
    eng.forward(torch.tensor([list(range(100,108))], device='cuda'), 0)
    torch.cuda.synchronize()
    print(f'RESULT prefill: marlin={calls["n"]} _prepared={calls["repack"]}')
    calls['n']=0; calls['repack']=0
    eng.forward(torch.tensor([[500]], device='cuda'), start_pos=8)
    torch.cuda.synchronize()
    print(f'RESULT decode : marlin={calls["n"]} _prepared={calls["repack"]}')
    calls['n']=0; calls['repack']=0
    eng.forward(torch.tensor([[501]], device='cuda'), start_pos=9)
    torch.cuda.synchronize()
    print(f'RESULT decode2: marlin={calls["n"]} _prepared={calls["repack"]}')