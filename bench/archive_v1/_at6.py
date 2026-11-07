import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.deepseek.attention import MLAWindowAttention
from tinyinfer.config import TinyConfig
from tinyinfer.weights import ShardStore
from transformers import AutoTokenizer
import pathlib
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
cfg = TinyConfig.from_hf(pathlib.Path(d))
store = ShardStore(d)
# 找一层 compress_ratio==0 的（window attention）
lid = next(i for i, r in enumerate(cfg.compress_ratios) if r == 0)
print('RESULT window layer:', lid, 'window:', cfg.sliding_window)
attn = MLAWindowAttention(store, lid, cfg)
attn.reset()
print('RESULT attn built')
# 逐步 decode vs 一次 prefill 的最后一行
torch.manual_seed(0)
T = 6
x = torch.randn(1, T, cfg.dim, dtype=torch.bfloat16) * 0.02
# A: 一次 prefill
ya = attn.forward(x, 0)
attn.reset()
# B: 逐步
outs = []
for t in range(T):
    y = attn.forward(x[:, t:t+1], t)
    outs.append(y)
yb = torch.cat(outs, dim=1)
d_ = (ya.float() - yb.float()).abs()
print(f'RESULT prefill vs stepwise: maxdiff={d_.max().item():.4e} rel={d_.max().item()/max(1e-6,ya.float().abs().max().item()):.2e}')