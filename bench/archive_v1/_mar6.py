import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.deepseek.gemv_fp4_marlin import gemv_fp4_fast
from tinyinfer.loading import ExpertStore
from tinyinfer.weights import ShardStore
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
store = ShardStore(d)
es = ExpertStore.from_store(store, cache_bytes=2<<30, device='cuda')
with es.experts(3, [3]) as recs:
    t = recs[0].tensors_on('cuda')
    w1, s1 = t['w1.weight'], t['w1.scale']
K = w1.shape[1]*2
x = torch.randn(1, K, dtype=torch.bfloat16, device='cuda')
print('RESULT direct call:')
try:
    out = gemv_fp4_fast(x, w1, s1)
    print('  ok', out.shape)
except Exception as e:
    import traceback; traceback.print_exc()
# 从 fp4_linear 里调
from tinyinfer.gemm import fp4_linear
print('RESULT via fp4_linear:')
try:
    out2 = fp4_linear(x, w1, s1)
    print('  ok', out2.shape)
except Exception as e:
    import traceback; traceback.print_exc()
# 手动看分支条件
print(f'RESULT cond: is_cuda={x.is_cuda} dim={x.dim()} size0={x.size(0)}')