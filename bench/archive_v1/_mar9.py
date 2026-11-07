import sys, torch, time
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.deepseek import fp4_native
from tinyinfer.deepseek.gemv_fp4_marlin import gemv_fp4_fast, _ops
from tinyinfer.loading import ExpertStore
from tinyinfer.weights import ShardStore
print('RESULT ops:', _ops() is not None)
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
store = ShardStore(d)
es = ExpertStore.from_store(store, cache_bytes=2<<30, device='cuda')
with es.experts(3, [3]) as recs:
    t = recs[0].tensors_on('cuda')
    w1, s1 = t['w1.weight'], t['w1.scale']
N, K2 = w1.shape; K = K2*2
bytes_w = N*K2 + N*(K//32)
print(f'RESULT matrix [N={N}, K={K}] = {bytes_w/2**20:.2f} MB')
# 固定 tensor 让 marlin repack 缓存命中
w1 = w1.clone(); s1 = s1.clone()
x = torch.randn(1, K, dtype=torch.bfloat16, device='cuda')
aq, asc = fp4_native.quantize_act(x, K)
def bench(fn, n=200, warm=20):
    for _ in range(warm): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize()
    return (time.perf_counter()-t0)/n
try:
    tm = bench(lambda: gemv_fp4_fast(x, w1, s1))
    print(f'RESULT marlin : {tm*1e6:7.1f} us  {bytes_w/tm/1e9:6.1f} GB/s')
except Exception as e:
    print('RESULT marlin FAILED', repr(e)[:110]); tm=None
tn = bench(lambda: fp4_native.gemv_fp4_pq(aq[0], asc[0], w1, s1))
print(f'RESULT native : {tn*1e6:7.1f} us  {bytes_w/tn/1e9:6.1f} GB/s')
if tm: print(f'RESULT marlin/native speed ratio: {tn/tm:.2f}x faster')
# 正确性对比
a = fp4_native.gemv_fp4_pq(aq[0], asc[0], w1, s1).float().flatten()
b = gemv_fp4_fast(x, w1, s1).float().flatten()
print(f'RESULT numeric maxdiff={(a-b).abs().max().item():.4e} mean|v|={a.abs().mean().item():.3f}')