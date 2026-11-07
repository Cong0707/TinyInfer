import sys, torch, collections
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=list(range(40)))
es = eng.expert_store
# 记录整段生成的专家访问序列
seq = []
orig = es._fetch_inner
def w(layer, experts, **kw):
    for e in experts: seq.append((int(layer), int(e)))
    return orig(layer, experts, **kw)
es._fetch_inner = w
with torch.inference_mode():
    eng.reset()
    ids = list(range(100, 108))
    out = list(eng.stream_generate(ids, max_new_tokens=6, temperature=0.0))
print(f'RESULT generated {len(out)} tokens, recorded {len(seq)} expert visits')
tokens_of = len(out) + 1
visits_per_tok = len(seq) / tokens_of
print(f'RESULT visits per token: {visits_per_tok:.0f}')
# 假设完美 LRU 缓存，模拟不同容量下的命中率
def sim(cap):
    cache = collections.OrderedDict()
    hit = 0
    for k in seq:
        if k in cache:
            hit += 1; cache.move_to_end(k)
        else:
            cache[k] = 1
            if len(cache) > cap: cache.popitem(last=False)
    return hit / len(seq)
for cap in (200, 400, 862, 2000, 5000):
    print(f'RESULT cap={cap:5d} entries ({cap*18.8/1024:5.1f} GB): hit={sim(cap)*100:5.1f}%')