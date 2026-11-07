import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=list(range(40)))
print(f'RESULT gpu_layers={eng.gpu_layers} stream={len(eng._stream_layers())}')
st = getattr(eng,'_flat_state',None)
if st:
    lay, bufs, host = st
    print(f'RESULT flat layers={len(lay)} ring bufs={len(bufs)}')
    if bufs: print(f'RESULT buf size={bufs[0].numel()/2**20:.0f} MB each')
    sizes = {k: v[1] for k,v in lay.items()}
    if sizes:
        ks = list(sizes)[:3]
        for k in ks: print(f'RESULT layer {k} slab need={sizes[k]/2**20:.1f} MB')
        print(f'RESULT sum of 30 layer sizes = {sum(sizes.values())/2**30:.2f} GB')