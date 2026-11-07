import sys, torch, time
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=list(range(40)))
es = eng.expert_store
# MoE 内部用已有 _parts 打点
with torch.inference_mode():
    eng.reset()
    eng.forward(torch.tensor([list(range(100,108))], device='cuda'), 0)
    torch.cuda.synchronize()
    for b in getattr(eng,'_blocks',{}).values():
        f = getattr(b,'ffn',None)
        if f is not None: f._parts = None
    torch.cuda.synchronize(); t0=time.perf_counter()
    eng.forward(torch.tensor([[500]], device='cuda'), start_pos=8)
    torch.cuda.synchronize(); total=(time.perf_counter()-t0)*1000
agg={}
for b in getattr(eng,'_blocks',{}).values():
    f=getattr(b,'ffn',None); p=getattr(f,'_parts',None) if f is not None else None
    if p:
        for k,v in p.items(): agg[k]=agg.get(k,0)+v
print(f'RESULT total={total:.0f}ms')
for k,v in sorted(agg.items(), key=lambda kv:-kv[1]):
    print(f'RESULT   {k:10s} {v*1000:6.0f}ms  {v*1000/total*100:4.0f}%')
s = sum(agg.values())
print(f'RESULT   sum {s*1000:.0f}ms, rest {total-s*1000:.0f}ms')