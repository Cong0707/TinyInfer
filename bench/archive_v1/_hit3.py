import sys, torch, collections
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=list(range(40)))
es = eng.expert_store
seq = []
_orig = es._materialize
def w(layer, expert, **kw):
    seq.append((layer, expert))
    return _orig(layer, expert, **kw)
es._materialize = w
with torch.inference_mode():
    eng.reset()
    eng.forward(torch.tensor([list(range(100,108))], device='cuda'), 0)
    torch.cuda.synchronize()
    seq.clear()
    for i in range(12):
        eng.forward(torch.tensor([[500+i]], device='cuda'), start_pos=8+i)
    torch.cuda.synchronize()
print(f'RESULT recorded {len(seq)} visits over 12 steps')
def sim(cap):
    cache = collections.OrderedDict(); hit = 0
    for k in seq:
        if k in cache: hit += 1; cache.move_to_end(k)
        else:
            cache[k] = 1
            if len(cache) > cap: cache.popitem(last=False)
    return hit/len(seq)
for cap in (400, 862, 1500, 3000, 6000):
    print(f'RESULT cap={cap:5d} ({cap*18.8/1024:5.1f} GB): {sim(cap)*100:5.1f}%')