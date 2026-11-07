import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.weights import ShardStore
from tinyinfer.deepseek import fp4_native
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
st = ShardStore(d)
L = 3
wg = st.get(f'layers.{L}.ffn.experts.0.w1.weight').to('cuda')
sg = st.get(f'layers.{L}.ffn.experts.0.w1.scale').to('cuda')
N, K2 = wg.shape; K = K2*2
x = torch.randn(1, K, dtype=torch.bfloat16, device='cuda')
aq, asc = fp4_native.quantize_act(x, K)
print('aq', tuple(aq.shape), aq.dtype, 'nan:', torch.isnan(aq.float()).any().item())
print('asc', tuple(asc.shape), asc.dtype, 'nan:', torch.isnan(asc).any().item())
print('asc sample', asc.flatten()[:4].tolist())
r = fp4_native.gemv_fp4_pq(aq[0], asc[0], wg, sg)
print('ref', tuple(r.shape), 'nan:', torch.isnan(r.float()).any().item(), 'sample', r.flatten()[:3].tolist())