import sys, torch, collections
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=list(range(40)))
import tinyinfer.loading as L
big = []
_a = L._async_h2d
import traceback
def w(dst, src, n):
    if n > 11_000_000 and n < 13_000_000:
        st = traceback.extract_stack()
        big.append(tuple(f'{f.name}:{f.lineno}' for f in st[-6:-2]))
    return _a(dst, src, n)
L._async_h2d = w
with torch.inference_mode():
    eng.reset()
    eng.forward(torch.tensor([list(range(100,108))], device='cuda'), 0)
    torch.cuda.synchronize()
    big.clear()
    eng.forward(torch.tensor([[500]], device='cuda'), start_pos=8)
    torch.cuda.synchronize()
c = collections.Counter(big)
print(f'RESULT 11.8MB origins: {len(c)} distinct')
for k,n in c.most_common(4):
    print(f'RESULT x{n}: {k}')