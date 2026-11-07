import sys, torch, collections
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=list(range(40)))
es = eng.expert_store
reads = []
_rb = es.reader.read_blob
import time
def wrb(blob):
    reads.append((blob.layer, blob.expert, time.perf_counter()))
    return _rb(blob)
es.reader.read_blob = wrb
with torch.inference_mode():
    eng.reset()
    eng.forward(torch.tensor([list(range(100,108))], device='cuda'), 0)
    torch.cuda.synchronize()
    reads.clear()
    eng.forward(torch.tensor([[500]], device='cuda'), start_pos=8)
    torch.cuda.synchronize()
c = collections.Counter((l,e) for l,e,_ in reads)
dups = [(k,v) for k,v in c.items() if v>1]
print(f'RESULT dups: {len(dups)}')
for (l,e),v in dups[:4]:
    ts = [t for L,E,t in reads if (L,E)==(l,e)]
    print(f'  layer={l} expert={e}: {v} reads, gap={max(ts)-min(ts):.3f}s')