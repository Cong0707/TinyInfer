import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=list(range(40)))
with torch.inference_mode():
    eng.reset()
    eng.forward(torch.tensor([list(range(100,108))], device='cuda'), 0)
    torch.cuda.synchronize()
st = getattr(eng, '_flat_state', None)
if st:
    lay, bufs, host = st
    print(f'RESULT flat_state: layers={len(lay)} bufs={len(bufs) if bufs else 0} host_slab={len(host) if host else 0}')
    if host: print(f'RESULT slab layers: {sorted(host)[:6]}')
    else: print('RESULT slab is None -> all layers take the per-tensor path')
print(f'RESULT _flat_applies: {getattr(eng,"_flat_applies",0)}')