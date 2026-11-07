import sys, torch, numpy as np
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.weights import ShardStore
from tinyinfer.deepseek.host_gate import route
from tinyinfer.deepseek.vllm_ops import gate_topk
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
st = ShardStore(d)
L = 3
w = st.get(f'layers.{L}.ffn.gate.weight').float()
b = st.get(f'layers.{L}.ffn.gate.bias').float()
x = torch.randn(1, 5120, dtype=torch.float32)
logits = x @ w.T
from tinyinfer.config import TinyConfig
cfg = TinyConfig.load(d) if hasattr(TinyConfig,'load') else None
print('route_scale', getattr(cfg,'route_scale',None), 'norm', getattr(cfg,'norm_topk_prob',None))
fused = gate_topk(logits, 6, True, 2.5, b)
gw, gi = fused[0], fused[1].long()
sel, idx = route(x[0].numpy(), w.numpy(), b.numpy(), 6, True, 2.5)
print('RESULT gpu idx', gi.flatten().tolist())
print('RESULT cpu idx', idx.flatten().tolist())
print('RESULT gpu w', gw.flatten().tolist()[:6])
print('RESULT cpu w', sel.flatten().tolist()[:6])