import sys, torch, collections
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=list(range(40)))
es = eng.expert_store
sr = es.reader
calls = []
_o = sr.read_many
def w(blobs, max_workers=None):
    calls.append(len(blobs))
    return _o(blobs, max_workers)
sr.read_many = w
_op = sr.read_many_pinned
def wp(blobs, max_workers=None):
    calls.append(('pinned', len(blobs)))
    return _op(blobs, max_workers)
sr.read_many_pinned = wp
with torch.inference_mode():
    eng.reset()
    eng.forward(torch.tensor([list(range(100,108))], device='cuda'), 0)
    torch.cuda.synchronize()
    calls.clear()
    eng.forward(torch.tensor([[500]], device='cuda'), start_pos=8)
    torch.cuda.synchronize()
c = collections.Counter(calls)
print(f'RESULT read_many calls: {len(calls)}, blob counts: {dict(list(c.items())[:8])}')
print(f'RESULT threads={sr.threads}')