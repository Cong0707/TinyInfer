import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.deepseek import fp4_native
from tinyinfer.deepseek.gemv_fp4_marlin import gemv_fp4_fast
from tinyinfer.loading import ExpertStore
from tinyinfer.weights import ShardStore
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
store = ShardStore(d)
es = ExpertStore.from_store(store, cache_bytes=2<<30, device='cuda')
layer = 3
with es.experts(layer, [3]) as recs:
    rec = recs[0]
    t = rec.tensors_on('cuda')
    w1, s1 = t['w1.weight'], t['w1.scale']
print('RESULT w1:', tuple(w1.shape), w1.dtype, ' scale:', tuple(s1.shape))
torch.manual_seed(0)
K = w1.shape[1] * 2
x = torch.randn(1, K, dtype=torch.bfloat16, device='cuda')
# native w4a8
aq, asc = fp4_native.quantize_act(x, K)
native = fp4_native.gemv_fp4_pq(aq, asc, w1, s1)
# marlin w4a16
try:
    marlin = gemv_fp4_fast(x, w1, s1)
    a = native.float().flatten(); b = marlin.float().flatten()
    diff = (a-b).abs()
    rel = (diff / a.abs().clamp(min=1e-6)).max().item()
    print(f'RESULT native[:4] {a[:4].tolist()}')
    print(f'RESULT marlin[:4] {b[:4].tolist()}')
    print(f'RESULT maxdiff={diff.max().item():.4e} max_rel={rel:.4e}')
except Exception as e:
    import traceback; traceback.print_exc()