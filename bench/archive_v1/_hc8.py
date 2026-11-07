import sys, torch, time
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.deepseek import hc
from tinyinfer.kernels import native
b, s, hcm = 1, 1, 4
mixes = torch.randn(b, s, hcm*hcm + 2*hcm, device='cuda')
scale = torch.rand(3, device='cuda') + 0.5
base = torch.randn(2*hcm + hcm*hcm, device='cuda')
def bench(fn,N=500):
    for _ in range(20): fn()
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(N): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t0)/N*1e6
t = bench(lambda: hc._native_sinkhorn(mixes, scale, base, hcm, 20, 1e-6))
print(f'RESULT _native_sinkhorn: {t:.2f} us  (was 103.00)')
comb = (mixes[..., 2*hcm:].view(b,s,hcm,hcm)*scale[2] + base[2*hcm:].view(hcm,hcm)).contiguous()
o2 = torch.empty_like(comb)
native.launch(native.kernel('sinkhorn','sinkhorn_hc'), (1,), (64,), [comb, o2, 1, hcm, 20, 1e-6])
torch.cuda.synchronize()
r = hc._native_sinkhorn(mixes, scale, base, hcm, 20, 1e-6)
print(f'RESULT comb identical: {torch.equal(r[2], o2)}')