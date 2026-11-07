import sys, torch, time
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=list(range(40)))
es = eng.expert_store
ram = es.ram
sr = ram.reader
cnt = {'pooled':0,'single':0,'slots_none':0}
_o = ram._read_many_pooled
def w(blobs, pool=None):
    cnt['pooled'] += 1
    return _o(blobs, pool)
ram._read_many_pooled = w
_rp = ram._read_pinned
def wp(b):
    cnt['single'] += 1
    return _rp(b)
ram._read_pinned = wp
with torch.inference_mode():
    eng.reset()
    eng.forward(torch.tensor([list(range(100,108))], device='cuda'), 0)
    torch.cuda.synchronize()
    cnt.update(pooled=0, single=0)
    eng.forward(torch.tensor([[500]], device='cuda'), start_pos=8)
    torch.cuda.synchronize()
print(f'RESULT decode: pooled={cnt["pooled"]} single={cnt["single"]}')
st = es.stats()
print(f'RESULT reader gbps: {st["reader"]["gbps"]:.2f}')