import sys, torch, pathlib
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.deepseek.attention import MLAWindowAttention
from tinyinfer.config import TinyConfig
from tinyinfer.weights import ShardStore
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
cfg = TinyConfig.from_hf(pathlib.Path(d))
store = ShardStore(d)
lid = next(i for i, r in enumerate(cfg.compress_ratios) if r == 0)
a = MLAWindowAttention(store, lid, cfg)
print('RESULT has to():', hasattr(a, 'to'))
attrs = [x for x in vars(a) if not x.startswith('_')]
print('RESULT attrs:', attrs[:16])
for name in attrs[:8]:
    v = getattr(a, name)
    if torch.is_tensor(v):
        print(f'  {name}: {tuple(v.shape)} {v.device} {v.dtype}')
    elif hasattr(v, 'weight'):
        print(f'  {name}: QuantLinear weight on {getattr(v.weight,"device","?")}')