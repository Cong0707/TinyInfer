import sys, torch, time
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.deepseek import fp4_native
from tinyinfer.deepseek.gemv_fp4_marlin import gemv_fp4_fast
from tinyinfer.loading import ExpertStore
from tinyinfer.weights import ShardStore
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
store = ShardStore(d)
es = ExpertStore.from_store(store, cache_bytes=2<<30, device='cuda')
with es.experts(3, [3]) as recs:
    t = recs[0].tensors_on('cuda')
    w1, s1 = t['w1.weight'], t['w1.scale']
K = w1.shape[1] * 2
N = w1.shape[0]
torch.manual_seed(0)
for M in (1, 2, 4, 8):
    x = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
    aq, asc = fp4_native.quantize_act(x, K)
    # native
    for _ in range(3):
        if M == 1: fp4_native.gemv_fp4_pq(aq[0], asc[0], w1, s1)
        else: [fp4_native.gemv_fp4_pq(aq[i], asc[i], w1, s1) for i in range(M)]
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(20):
        if M == 1: fp4_native.gemv_fp4_pq(aq[0], asc[0], w1, s1)
        else: [fp4_native.gemv_fp4_pq(aq[i], asc[i], w1, s1) for i in range(M)]
    torch.cuda.synchronize(); tn=(time.perf_counter()-t0)/20*1000
    # marlin
    for _ in range(3): gemv_fp4_fast(x, w1, s1)
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(20): gemv_fp4_fast(x, w1, s1)
    torch.cuda.synchronize(); tm=(time.perf_counter()-t0)/20*1000
    print(f'RESULT M={M}: native={tn:.3f}ms marlin={tm:.3f}ms ratio={tm/tn:.2f}x')