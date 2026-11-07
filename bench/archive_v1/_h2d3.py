import sys, torch, collections
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=list(range(40)))
import tinyinfer.loading as L
calls = []
_a = L._async_h2d
def w(dst, src, n):
    calls.append(n)
    return _a(dst, src, n)
L._async_h2d = w
with torch.inference_mode():
    eng.reset()
    eng.forward(torch.tensor([list(range(100,108))], device='cuda'), 0)
    torch.cuda.synchronize()
    calls.clear()
    eng.forward(torch.tensor([[500]], device='cuda'), start_pos=8)
    torch.cuda.synchronize()
c = collections.Counter(calls)
print(f'RESULT distinct sizes: {len(c)}')
for sz, n in c.most_common(8):
    print(f'RESULT   {sz/1e6:8.2f} MB x {n:4d} = {sz*n/1e9:.2f} GB')
print(f'RESULT total {sum(calls)/1e9:.2f} GB in {len(calls)} calls')