import sys, torch, collections
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=list(range(40)))
import tinyinfer.loading as L
cnt = collections.Counter()
_a = L._async_h2d
def w(dst, src, n):
    cnt['h2d'] += 1; cnt['bytes'] += n
    return _a(dst, src, n)
L._async_h2d = w
with torch.inference_mode():
    eng.reset()
    eng.forward(torch.tensor([list(range(100,108))], device='cuda'), 0)
    torch.cuda.synchronize()
    cnt.clear()
    eng.forward(torch.tensor([[500]], device='cuda'), start_pos=8)
    torch.cuda.synchronize()
print(f'RESULT _async_h2d calls={cnt["h2d"]} bytes={cnt["bytes"]/1e9:.2f} GB avg={cnt["bytes"]/max(1,cnt["h2d"])/1e6:.2f} MB')