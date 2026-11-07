import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
import tinyinfer.deepseek.gemv_fp4_marlin as MAR
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=list(range(40)))
calls = {'marlin': 0}
orig = MAR.gemv_fp4_fast
def w(*a, **kw):
    calls['marlin'] += 1
    return orig(*a, **kw)
MAR.gemv_fp4_fast = w
# 同时也 patch gemm 里的引用
import tinyinfer.gemm as G
ids = list(range(100, 104))
with torch.inference_mode():
    eng.reset()
    eng.forward(torch.tensor([ids], device='cuda'), 0)   # prefill
    torch.cuda.synchronize()
    print(f'RESULT after prefill(4 tok): marlin_calls={calls["marlin"]}')
    calls['marlin'] = 0
    eng.forward(torch.tensor([[500]], device='cuda'), start_pos=4)  # 1 decode step
    torch.cuda.synchronize()
    print(f'RESULT after 1 decode step: marlin_calls={calls["marlin"]}')