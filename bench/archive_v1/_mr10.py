import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
import tinyinfer.deepseek.gemv_fp4_marlin as MM
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=list(range(40)))
calls = [0]
orig = MM.MarlinFp4Weight.__init__
def patched(self, w, s):
    calls[0] += 1
    return orig(self, w, s)
MM.MarlinFp4Weight.__init__ = patched
with torch.inference_mode():
    eng.reset()
    eng.forward(torch.tensor([list(range(100,108))], device='cuda'), 0)
    torch.cuda.synchronize()
    calls[0] = 0
    eng.forward(torch.tensor([[500]], device='cuda'), start_pos=8)
    torch.cuda.synchronize()
print(f'RESULT repacks per decode step: {calls[0]}')
print(f'RESULT 理论专家数: 240（6 x 40 层）')