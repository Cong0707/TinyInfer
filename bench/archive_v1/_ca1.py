import sys, torch, pathlib
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.deepseek.compressed_attention import MLACompressedAttention
from tinyinfer.config import TinyConfig
from tinyinfer.weights import ShardStore
import os
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
cfg = TinyConfig.from_hf(pathlib.Path(d))
store = ShardStore(d)
# 找一层 compress_ratio != 0
lid = next(i for i, r in enumerate(cfg.compress_ratios) if r != 0)
print('RESULT compressed layer:', lid, 'ratio:', cfg.compress_ratios[lid])
a = MLACompressedAttention(store, lid, cfg)
for name, v in list(vars(a).items()):
    if name.startswith('_'): continue
    if torch.is_tensor(v): setattr(a, name, v.to('cuda'))
    elif hasattr(v, 'to'):
        try: v.to('cuda')
        except Exception: pass
a.reset()
torch.manual_seed(0)
T = 6
x = torch.randn(1, T, cfg.dim, dtype=torch.bfloat16, device='cuda') * 0.02
try:
    ya = a.forward(x, 0)
    a.reset()
    outs = [a.forward(x[:, t:t+1], t) for t in range(T)]
    yb = torch.cat(outs, dim=1)
    d_ = (ya.float()-yb.float()).abs()
    print(f'RESULT prefill vs stepwise: maxdiff={d_.max().item():.4e} rel={d_.max().item()/max(1e-6,ya.float().abs().max().item()):.2e}')
except Exception as e:
    import traceback; traceback.print_exc()