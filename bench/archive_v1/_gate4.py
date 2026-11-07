import sys, torch, numpy as np
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.weights import ShardStore
from tinyinfer.deepseek.host_gate import route
import torch.nn.functional as F
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
st = ShardStore(d)
L = 3
w = st.get(f'layers.{L}.ffn.gate.weight').float()
b = st.get(f'layers.{L}.ffn.gate.bias').float()
torch.manual_seed(0)
# 用真实量级的输入
x = torch.randn(1, 5120, dtype=torch.float32) * 0.1
# GPU 路径（复刻 moe.py 的 torch 分支）
logits = F.linear(x, w)
scores = F.softplus(logits).sqrt()
ranking = scores + b
order = torch.argsort(ranking, dim=-1, descending=True, stable=True)
gi = order[..., :6]
gw = scores.gather(1, gi)
gw = gw / (gw.sum(dim=-1, keepdim=True) + 1e-20)
# 找 route_scale
from tinyinfer.config import TinyConfig
import json, pathlib
cfgj = json.load(open(pathlib.Path(d)/'config.json'))
tc = cfgj.get('text_config', cfgj)
rs = tc.get('route_scale', 1.0)
print('route_scale', rs, 'norm_topk', tc.get('norm_topk_prob'))
gw = gw * rs
# CPU
from tinyinfer.deepseek.host_gate import route as R
sel, idx = R(x[0].numpy(), w.numpy(), b.numpy(), 6, True, rs)
print('RESULT gpu idx', gi.flatten().tolist())
print('RESULT cpu idx', idx.flatten().tolist())
print('RESULT gpu w', [round(v,5) for v in gw.flatten().tolist()])
print('RESULT cpu w', [round(float(v),5) for v in sel.flatten().tolist()])