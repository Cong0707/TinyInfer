import sys, torch, time
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=list(range(40)))
T = {}
def tk(k, dt): T[k] = T.get(k,0.0) + dt
# 打点：attention / hc / ffn
import tinyinfer.deepseek.model as MD
orig = MD.Block.forward
def wb(self, x, start_pos, pre_mix, positions=None):
    t0=time.perf_counter()
    r = orig(self, x, start_pos, pre_mix, positions)
    tk('block_total', time.perf_counter()-t0)
    return r
MD.Block.forward = wb
with torch.inference_mode():
    eng.reset()
    eng.forward(torch.tensor([list(range(100,108))], device='cuda'), 0)
    torch.cuda.synchronize()
    T.clear()
    torch.cuda.synchronize(); t0=time.perf_counter()
    eng.forward(torch.tensor([[500]], device='cuda'), start_pos=8)
    torch.cuda.synchronize(); total=(time.perf_counter()-t0)*1000
print(f'RESULT total={total:.0f}ms')
for k,v in sorted(T.items(), key=lambda kv:-kv[1]):
    print(f'RESULT   {k:12s} {v*1000:6.0f}ms {v*1000/total*100:4.0f}%')
print(f'RESULT   outside     {total-sum(T.values())*1000:6.0f}ms')