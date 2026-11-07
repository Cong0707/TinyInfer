import sys, torch, os
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=list(range(40)))
es = eng.expert_store
per_call = []
orig = es._fetch_inner
def w(layer, experts, **kw):
    ex = [int(e) for e in experts]
    per_call.append((layer, len(ex)))
    return orig(layer, experts, **kw)
es._fetch_inner = w
for n in (1, 8):
    ids = list(range(100, 100+n))
    per_call.clear()
    with torch.inference_mode():
        eng.reset()
        list(eng.stream_generate(ids, max_new_tokens=2, temperature=0.0))
        per_call.clear()
        list(eng.stream_generate(ids, max_new_tokens=1, temperature=0.0))
    tot = sum(c for _,c in per_call)
    by = {}
    for l,c in per_call: by[l] = by.get(l,0)+c
    print(f'RESULT len={n}: calls={len(per_call)} experts={tot} per_layer={list(by.items())[:5]}')