import sys, torch, time, collections
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=list(range(40)))
es = eng.expert_store
visits = collections.Counter()
misses = collections.Counter()
_orig = es._materialize
def w(layer, expert, **kw):
    visits[(layer, expert)] += 1
    if kw.get('data') is None and not es.ram.contains(layer, expert):
        misses[(layer, expert)] += 1
    return _orig(layer, expert, **kw)
es._materialize = w
with torch.inference_mode():
    eng.reset()
    eng.forward(torch.tensor([list(range(100,108))], device='cuda'), 0)
    torch.cuda.synchronize()
    visits.clear(); misses.clear()
    eng.forward(torch.tensor([[500]], device='cuda'), start_pos=8)
    torch.cuda.synchronize()
uniq = len(visits)
tot = sum(visits.values())
print(f'RESULT decode: visits={tot} distinct={uniq} misses={len(misses)}')
print(f'RESULT hit rate {(1-len(misses)/max(1,uniq))*100:.1f}%')
print(f'RESULT bytes if all miss: {uniq*18.8/1000:.2f} GB, actual misses: {len(misses)*18.8/1000:.2f} GB')