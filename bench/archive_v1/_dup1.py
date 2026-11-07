import sys, torch, collections
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=list(range(40)))
es = eng.expert_store
reads = collections.Counter()
orig = es._fetch_inner
def w(layer, experts, **kw):
    ex = [int(e) for e in experts]
    have = es.ram.contains_many(layer, ex)
    for e in ex:
        if not have.get(e): reads[(layer, e)] += 1
    return orig(layer, experts, **kw)
es._fetch_inner = w
with torch.inference_mode():
    eng.reset()
    eng.forward(torch.tensor([list(range(100,108))], device='cuda'), 0)
    torch.cuda.synchronize()
    reads.clear()
    eng.forward(torch.tensor([[500]], device='cuda'), start_pos=8)
    torch.cuda.synchronize()
per_layer = collections.Counter()
for (l,e),c in reads.items(): per_layer[l] += c
tot = sum(reads.values())
dup = sum(c-1 for c in reads.values())
print(f'RESULT decode step: distinct expert-misses={len(reads)} total_misses={tot} duplicates={dup}')
print(f'RESULT bytes if each read once: {len(reads)*18.8/1000:.2f} GB')
print(f'RESULT bytes with duplicates:   {tot*18.8/1000:.2f} GB')
print(f'RESULT per-layer miss counts: {sorted(per_layer.items())[:8]}')