import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.weights import ShardStore
from tinyinfer.deepseek.gemv_fp4 import gemv_fp4_pq, quantize_act, _scale
from tinyinfer.deepseek.gemv_fp4_marlin import gemv_fp4_fast, available
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
st = ShardStore(d)
name = 'layers.3.ffn.experts.0.w1.weight'
w = st.get(name)
print('w', w.shape, w.dtype, 'marlin available:', available())
K = w.shape[1]*2
x = torch.randn(1, K, dtype=torch.bfloat16, device='cuda')
g = st.get('layers.3.ffn.experts.0.w1.scale').to('cuda')
aq, asc = quantize_act(x, K)
aq = aq.reshape(-1); asc = asc.reshape(-1)
y_pq = gemv_fp4_pq(aq, asc, w.to('cuda'), g)
y_ml = gemv_fp4_fast(x, w.to('cuda'), g)
a = y_pq.float().flatten(); b = y_ml.float().flatten()
print(f'RESULT maxdiff={ (a-b).abs().max().item():.4e}  rel={(a-b).abs().max().item()/max(1e-6,a.abs().max().item()):.3e}')
print('first4 pq', y_pq.flatten()[:4].tolist())
print('first4 ml', y_ml.flatten()[:4].tolist())