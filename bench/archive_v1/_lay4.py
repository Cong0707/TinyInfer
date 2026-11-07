import sys, torch, os
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
os.environ['TI_MOE_PROF'] = '1'
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=list(range(40)))
def run(n, label):
    ids = list(range(100, 100+n))
    with torch.inference_mode():
        eng.reset()
        list(eng.stream_generate(ids, max_new_tokens=2, temperature=0.0))
        for b in getattr(eng,'_blocks',{}).values():
            f = getattr(b, 'ffn', None)
            if f is not None: f._parts = None
        list(eng.stream_generate(ids, max_new_tokens=1, temperature=0.0))
    agg = {}
    for b in getattr(eng,'_blocks',{}).values():
        f = getattr(b, 'ffn', None)
        p = getattr(f, '_parts', None) if f is not None else None
        if p:
            for k,v in p.items(): agg[k]=agg.get(k,0)+v
    print(f'RESULT {label}: ' + '  '.join(f'{k}={v*1000:.0f}ms' for k,v in sorted(agg.items())))
run(1, 'len=1')
run(8, 'len=8')