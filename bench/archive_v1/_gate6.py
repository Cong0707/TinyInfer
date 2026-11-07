import sys, torch, numpy as np
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.weights import ShardStore
from tinyinfer.deepseek.host_gate import route as R
import torch.nn.functional as F
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
st = ShardStore(d)
L = 0
w = st.get(f'layers.{L}.ffn.gate.weight').float()
b = st.get(f'layers.{L}.ffn.gate.bias').float()
# 真实量级
torch.manual_seed(0)
x = torch.randn(1, 5120, dtype=torch.float32) * 0.02
logits = F.linear(x, w)
scores = F.softplus(logits).sqrt()
ranking = scores + b
order = torch.argsort(ranking, dim=-1, descending=True, stable=True)
gi = order[..., :6]
gw = scores.gather(1, gi); gw = gw/(gw.sum(-1,keepdim=True)+1e-20); gw = gw*1.5
sel, idx = R(x[0].numpy(), w.numpy(), b.numpy(), 6, True, 1.5)
print('RESULT gpu idx', gi.flatten().tolist())
print('RESULT cpu idx', idx.flatten().tolist())
print('RESULT gpu w', [round(float(v),6) for v in gw.flatten().tolist()])
print('RESULT cpu w', [round(float(v),6) for v in sel.flatten().tolist()])
print('RESULT w match', np.allclose(gw.flatten().numpy(), sel.flatten(), atol=1e-6))