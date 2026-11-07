import sys, torch, time
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.weights import ShardStore
from tinyinfer.kernels import native
from tinyinfer.kernels.quant import act_quant
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
store = ShardStore(d)
# 共享专家的三个矩阵
mats = []
for nm in ('w1','w3','w2'):
    k = f'layers.3.ffn.shared_experts.{nm}.weight'
    s = f'layers.3.ffn.shared_experts.{nm}.scale'
    if store.has(k):
        w = store.get(k).cuda(); ws = store.get(s).cuda()
        mats.append((nm, w, ws))
print('RESULT shared mats:', [(n, tuple(w.shape)) for n,w,_ in mats])
x = torch.randn(1, 5120, dtype=torch.bfloat16, device='cuda')
aq, asc = act_quant(x, 32, 'ue8m0', torch.float8_e8m0fnu)
a = aq.view(torch.uint8).reshape(-1); sc = asc.view(torch.uint8).reshape(-1)
def bench(f, n=300):
    for _ in range(30): f()
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.perf_counter()-t0)/n*1e6
# 逐个 launch
fns = []
for nm, w, ws in mats:
    N, K = w.shape
    y = torch.empty(N, dtype=torch.bfloat16, device='cuda')
    fn = native.BoundLaunch(native.kernel('fp8_gemv_bs','fp8_gemv_bs'), ((N+15)//16,), (512,), 'TTTTOii')
    fns.append((fn, w.view(torch.uint8).reshape(-1), ws.view(torch.uint8).reshape(-1), y, N, K))
def seq():
    for fn, wb, wsb, y, N, K in fns:
        fn(a, sc, wb, wsb, y, N, K, stream=None)
t = bench(seq)
print(f'RESULT {len(fns)} launches: {t:.2f} us  ({t/len(fns):.1f} us each)')
print(f'RESULT driver floor would be {len(fns)*16.5:.1f} us')