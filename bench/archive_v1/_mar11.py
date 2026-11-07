import sys, torch, time
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.deepseek.gemv_fp4_marlin import gemv_fp4_fast, _prepared, MarlinFp4Weight
from tinyinfer.loading import ExpertStore
from tinyinfer.weights import ShardStore
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
store = ShardStore(d)
es = ExpertStore.from_store(store, cache_bytes=2<<30, device='cuda')
with es.experts(3, [3]) as recs:
    t = recs[0].tensors_on('cuda')
    w1, s1 = t['w1.weight'], t['w1.scale']
N, K2 = w1.shape; K = K2*2
x = torch.randn(1, K, dtype=torch.bfloat16, device='cuda')
def bench(fn, n=100, warm=10):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t0)/n
# 命中缓存（同一个 tensor 对象）
w1f, s1f = w1.clone(), s1.clone()
t_hit = bench(lambda: gemv_fp4_fast(x, w1f, s1f))
print(f'RESULT cache-hit  : {t_hit*1e6:7.1f} us')
# 每次新 tensor（模拟 data_ptr 变化）
def cold():
    w = w1.clone(); s = s1.clone()
    return gemv_fp4_fast(x, w, s)
t_cold = bench(cold, n=30, warm=3)
print(f'RESULT cache-miss : {t_cold*1e6:7.1f} us  (repack cost = {(t_cold-t_hit)*1e6:.1f} us)')
print(f'RESULT repack is {t_cold/t_hit:.1f}x the hit cost')