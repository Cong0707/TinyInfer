import sys, torch, time
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.weights import ShardStore
from tinyinfer import gemm
from tinyinfer.kernels import native
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
store = ShardStore(d)
w = store.get('layers.3.ffn.shared_experts.w1.weight').cuda()
ws = store.get('layers.3.ffn.shared_experts.w1.scale').cuda()
N, K = w.shape
x = torch.randn(1, K, dtype=torch.bfloat16, device='cuda')
def bench(fn,n=500):
    for _ in range(30): fn()
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t0)/n*1e6
print(f'RESULT fp8_linear total    : {bench(lambda: gemm.fp8_linear(x, w, ws)):.2f} us')
from tinyinfer.kernels.quant import act_quant
print(f'RESULT _act_quant          : {bench(lambda: act_quant(x, 32, "ue8m0", torch.float8_e8m0fnu)):.2f} us')
print(f'RESULT torch.empty y       : {bench(lambda: torch.empty(N, device="cuda", dtype=torch.bfloat16)):.2f} us')
print(f'RESULT w.view(uint8)       : {bench(lambda: w.view(torch.uint8).reshape(-1)):.2f} us')
print(f'RESULT ws.view(uint8)      : {bench(lambda: ws.view(torch.uint8).reshape(-1)):.2f} us')
aq, asc = act_quant(x, 32, 'ue8m0', torch.float8_e8m0fnu)
y = torch.empty(N, device='cuda', dtype=torch.bfloat16)
wb = w.view(torch.uint8).reshape(-1); wsb = ws.view(torch.uint8).reshape(-1)
a = aq.view(torch.uint8).reshape(-1); s = asc.view(torch.uint8).reshape(-1)
fn = native.BoundLaunch(native.kernel('fp8_gemv_bs','fp8_gemv_bs'), ((N+15)//16,), (512,), 'TTTTOii')
print(f'RESULT bound kernel only   : {bench(lambda: fn(a, s, wb, wsb, y, N, K, stream=None)):.2f} us')