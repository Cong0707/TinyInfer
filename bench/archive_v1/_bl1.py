import sys, torch, time
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.kernels import native
from tinyinfer.weights import ShardStore
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
store = ShardStore(d)
w = store.get('layers.3.ffn.shared_experts.w1.weight').cuda()
ws = store.get('layers.3.ffn.shared_experts.w1.scale').cuda()
N, K = w.shape
from tinyinfer.kernels.quant import act_quant
x = torch.randn(1, K, dtype=torch.bfloat16, device='cuda')
aq, asc = act_quant(x, 32, 'ue8m0', torch.float8_e8m0fnu)
y = torch.empty(N, device='cuda', dtype=torch.bfloat16)
a = aq.view(torch.uint8).reshape(-1); s = asc.view(torch.uint8).reshape(-1)
wb = w.view(torch.uint8).reshape(-1); wsb = ws.view(torch.uint8).reshape(-1)
fn = native.BoundLaunch(native.kernel('fp8_gemv_bs','fp8_gemv_bs'), ((N+15)//16,), (512,), 'TTTTOii')
def bench(f, n=500):
    for _ in range(50): f()
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.perf_counter()-t0)/n*1e6
print(f'RESULT bound+stream=None : {bench(lambda: fn(a, s, wb, wsb, y, N, K, stream=None)):.2f} us')
import ctypes
st = ctypes.c_void_p(int(torch.cuda.current_stream().cuda_stream))
print(f'RESULT bound+given stream: {bench(lambda: fn(a, s, wb, wsb, y, N, K, stream=st)):.2f} us')
# 纯 GPU 时间
torch.cuda.synchronize(); t0=time.perf_counter()
for _ in range(500): fn(a, s, wb, wsb, y, N, K, stream=st)
sub=(time.perf_counter()-t0)/500
torch.cuda.synchronize(); wall=(time.perf_counter()-t0)/500
print(f'RESULT submit-only: {sub*1e6:.2f} us')