import sys, torch, os
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=list(range(40)))
hp = eng.host_pool
print(f'RESULT after init: used={hp.used/2**30:.2f} free={hp.free}')
with torch.inference_mode():
    eng.reset()
    eng.forward(torch.tensor([list(range(100,108))], device='cuda'), 0)
    torch.cuda.synchronize()
print(f'RESULT after first fwd: used={hp.used/2**30:.2f} GB')
# 看 flat 相关
for a in ('_flat_blocks','_flat_pool','_slab'):
    v = getattr(eng, a, None)
    if v is not None:
        try: print(f'RESULT {a}: {len(v)} entries')
        except Exception: print(f'RESULT {a}: {v}')
print('RESULT es._pool:', getattr(eng.expert_store,'_pool',None))