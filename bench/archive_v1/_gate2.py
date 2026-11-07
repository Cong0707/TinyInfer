import sys, torch, time
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.weights import ShardStore
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
st = ShardStore(d)
L = 3
w = st.get(f'layers.{L}.ffn.gate.weight').float()
b = st.get(f'layers.{L}.ffn.gate.bias').float()
print('RESULT w', tuple(w.shape), 'b', tuple(b.shape))
# CPU 侧：转 numpy float32
wn = w.numpy(); bn = b.numpy()
x = torch.randn(1, w.shape[1], dtype=torch.float32)
xn = x.numpy()
# GPU 参考
import torch.nn.functional as F
logits_g = x @ w.T
scores_g = F.softplus(logits_g).sqrt()
np_ = torch
# CPU 实现
import numpy as np
def cpu_gate(xn, wn, bn, topk):
    logits = xn @ wn.T
    # softplus then sqrt, fp32
    sp = np.log1p(np.exp(-np.abs(logits))) + np.maximum(logits, 0)
    scores = np.sqrt(sp)
    s = scores + bn
    idx = np.argsort(-s, axis=-1)[:, :topk]
    sel = np.take_along_axis(s, idx, axis=-1)
    sel = sel / sel.sum(axis=-1, keepdims=True)
    return sel, idx
t0 = time.perf_counter()
for _ in range(50):
    wsel, widx = cpu_gate(xn.astype(np.float32), wn, bn, 6)
t = (time.perf_counter()-t0)/50
print(f'RESULT CPU gate+topk {t*1000:.2f} ms/层 -> 40层 {40*t*1000:.0f} ms')
# 对比数值
g_scores = scores_g + b
ref_idx = torch.argsort(-g_scores, dim=-1)[:, :6]
print('RESULT idx match:', (ref_idx.numpy() == widx).all())