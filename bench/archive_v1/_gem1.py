import sys, torch, time
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.weights import ShardStore
from tinyinfer import gemm
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
store = ShardStore(d)
w = store.get('layers.3.ffn.shared_experts.w1.weight').cuda()
ws = store.get('layers.3.ffn.shared_experts.w1.scale').cuda()
N, K = w.shape
x = torch.randn(1, K, dtype=torch.bfloat16, device='cuda')
def bench(fn,N2=200):
    for _ in range(20): fn()
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(N2): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t0)/N2*1e6
t = bench(lambda: gemm.fp8_linear(x, w, ws))
print(f'RESULT fp8_linear: {t:.2f} us')
y = gemm.fp8_linear(x, w, ws)
print(f'RESULT shape {tuple(y.shape)} finite={torch.isfinite(y.float()).all().item()}')