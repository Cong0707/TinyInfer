import sys, torch, collections
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=list(range(40)))
es = eng.expert_store; ram = es.ram
_orig = ram.get
def noget(layer, expert, **kw):
    kw['admit'] = False
    return _orig(layer, expert, **kw)
ram.get = noget
reads = collections.Counter()
_rb = es.reader.read_blob
def wrb(blob):
    reads[(blob.layer, blob.expert)] += 1
    return _rb(blob)
es.reader.read_blob = wrb
with torch.inference_mode():
    eng.reset()
    eng.forward(torch.tensor([list(range(100,108))], device='cuda'), 0)
    torch.cuda.synchronize()
    reads.clear()
    eng.forward(torch.tensor([[500]], device='cuda'), start_pos=8)
    torch.cuda.synchronize()
tot = sum(reads.values()); uniq = len(reads); dup = tot-uniq
print(f'RESULT total reads={tot} distinct={uniq} duplicates={dup}')
print(f'RESULT bytes: {tot*18.8/1000:.2f} GB (theory 240 experts = 4.51 GB)')
if dup: print(f'RESULT top dups: {[(k,v) for k,v in reads.most_common(3) if v>1]}')