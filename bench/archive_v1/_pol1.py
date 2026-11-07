import sys, torch, collections
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=list(range(40)))
es = eng.expert_store; ram = es.ram
seq = []
_o = ram.get
def w(layer, expert, **kw):
    seq.append((layer, expert))
    return _o(layer, expert, **kw)
ram.get = w
with torch.inference_mode():
    eng.reset()
    eng.forward(torch.tensor([list(range(100,108))], device='cuda'), 0)
    torch.cuda.synchronize()
    seq.clear()
    for i in range(12):
        eng.forward(torch.tensor([[500+i]], device='cuda'), start_pos=8+i)
    torch.cuda.synchronize()
st = es.stats()
print(f'RESULT visits recorded: {len(seq)} over 12 steps')
print(f'RESULT actual hit_rate: {st["ram"]["hit_rate"]:.3f}, entries {st["ram"]["entries"]}, policy {st["ram"]["policy"]}')
def sim(cap):
    c = collections.OrderedDict(); hit = 0
    for k in seq:
        if k in c: hit += 1; c.move_to_end(k)
        else:
            c[k] = 1
            if len(c) > cap: c.popitem(last=False)
    return hit/len(seq)
cap = st['ram']['entries']
print(f'RESULT perfect-LRU at cap={cap}: {sim(cap)*100:.1f}%')
print(f'RESULT gap: {(sim(cap)-st["ram"]["hit_rate"])*100:+.1f} points')