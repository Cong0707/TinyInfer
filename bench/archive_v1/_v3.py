import sys, torch, time, shutil, pathlib
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
build = pathlib.Path(r'E:\Develop\Deepseek\tinyinfer\src\tinyinfer\kernels\build')
orig = build / 'gemv_fp4_w4a8.cubin'
shutil.copy2(r'E:\temp\gemv_v3.cubin', orig)
from tinyinfer.deepseek import fp4_native
from tinyinfer.loading import ExpertStore
from tinyinfer.weights import ShardStore
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
store = ShardStore(d)
es = ExpertStore.from_store(store, cache_bytes=2<<30, device='cuda')
with es.experts(3, [3]) as recs:
    t = recs[0].tensors_on('cuda')
    w1, s1 = t['w1.weight'], t['w1.scale']
N, K2 = w1.shape; K = K2*2
bytes_w = N*K2 + N*(K//32)
x = torch.randn(1, K, dtype=torch.bfloat16, device='cuda')
aq, asc = fp4_native.quantize_act(x, K)
lut = torch.tensor([0.,0.5,1.,1.5,2.,3.,4.,6., -0.,-0.5,-1.,-1.5,-2.,-3.,-4.,-6.], device='cuda')
pk = w1.view(torch.uint8)
lo = (pk & 0x0F).long(); hi = ((pk >> 4) & 0x0F).long()
codes = torch.stack((lo, hi), dim=-1).reshape(N, K)
wf = lut[codes]
af = aq[0].view(torch.float8_e4m3fn).float().reshape(-1, 32)
nb = K // 32
asf = asc[0].float().flatten()[:nb]; wsf = s1.float().reshape(N, nb)
ref = torch.zeros(N, device='cuda')
for kb in range(nb):
    ref += (wf[:, kb*32:(kb+1)*32] * af[kb][None,:]).sum(1) * asf[kb] * wsf[:, kb]
out = fp4_native.gemv_fp4_pq(aq[0], asc[0], w1, s1).float().flatten()
diff=(out-ref).abs()
print(f'RESULT correctness: maxabs={diff.max().item():.4e} rel={diff.max().item()/ref.abs().max().item():.2e}')
for _ in range(10): fp4_native.gemv_fp4_pq(aq[0], asc[0], w1, s1)
torch.cuda.synchronize(); t0=time.perf_counter()
for _ in range(200): fp4_native.gemv_fp4_pq(aq[0], asc[0], w1, s1)
torch.cuda.synchronize(); dt=(time.perf_counter()-t0)/200
print(f'RESULT v3: {dt*1e6:.1f} us  {bytes_w/dt/1e9:.1f} GB/s')
print(f'RESULT v2 was 133.5 us / 46.9 GB/s')