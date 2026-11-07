import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
import tinyinfer.deepseek.gemv_fp4_marlin as MAR
calls = {'n':0, 'fail':0}
_o = MAR.gemv_fp4_fast
def w(*a, **kw):
    try:
        calls['n'] += 1
        return _o(*a, **kw)
    except Exception:
        calls['fail'] += 1; raise
MAR.gemv_fp4_fast = w
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=list(range(40)))
with torch.inference_mode():
    eng.reset()
    eng.forward(torch.tensor([list(range(100,108))], device='cuda'), 0)
    torch.cuda.synchronize()
    print(f'RESULT prefill: marlin calls={calls["n"]} fail={calls["fail"]}')
    calls['n']=0; calls['fail']=0
    eng.forward(torch.tensor([[500]], device='cuda'), start_pos=8)
    torch.cuda.synchronize()
    print(f'RESULT decode : marlin calls={calls["n"]} fail={calls["fail"]}')