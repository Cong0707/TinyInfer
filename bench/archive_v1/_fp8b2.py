import sys, torch, time
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.kernels import native
from tinyinfer.weights import ShardStore
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
store = ShardStore(d)
w = store.get('layers.3.ffn.shared_experts.w1.weight').cuda()
ws = store.get('layers.3.ffn.shared_experts.w1.scale').cuda()
N, K = w.shape
print(f'RESULT w [{N},{K}] {w.dtype} = {w.numel()/2**20:.2f} MB, ws {tuple(ws.shape)} {ws.dtype}')
from tinyinfer.kernels.quant import act_quant
x = torch.randn(1, K, dtype=torch.bfloat16, device='cuda')
aq, asc = act_quant(x, 32, 'ue8m0', torch.float8_e8m0fnu)
aq = aq[0].view(torch.uint8).contiguous()
asc = asc[0].view(torch.uint8).contiguous()
wb = w.view(torch.uint8).contiguous()
wsb = ws.view(torch.uint8).contiguous()
y = torch.empty(N, dtype=torch.bfloat16, device='cuda')
k = native.kernel('fp8_gemv_bs', 'fp8_gemv_bs')
def run():
    native.launch(k, ((N + 15)//16,), (512,), [aq, asc, wb, wsb, y, N, K])
for _ in range(20): run()
torch.cuda.synchronize(); t0=time.perf_counter()
for _ in range(200): run()
torch.cuda.synchronize(); dt=(time.perf_counter()-t0)/200
bytes_w = wb.numel() + wsb.numel()
print(f'RESULT fp8_gemv_bs: {dt*1e6:.1f} us  {bytes_w/dt/1e9:.1f} GB/s')
print(f'RESULT upper bound 295 GB/s would be {bytes_w/295e9*1e6:.1f} us')