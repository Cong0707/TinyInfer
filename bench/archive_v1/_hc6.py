import sys, torch, time
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.deepseek import hc
b, s, hcm = 1, 1, 4
mixes = torch.randn(b, s, hcm*hcm + 2*hcm, device='cuda')
scale = torch.rand(3, device='cuda') + 0.5
base = torch.randn(2*hcm + hcm*hcm, device='cuda')
def bench(fn,N=500):
    for _ in range(20): fn()
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(N): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t0)/N*1e6
print(f'RESULT _native_sinkhorn: {bench(lambda: hc._native_sinkhorn(mixes, scale, base, hcm, 20, 1e-6)):.2f} us')
print(f'RESULT _sinkhorn_body  : {bench(lambda: hc._sinkhorn_body(mixes, scale, base, hcm, 20, 1e-6)):.2f} us')
# 分解
hc_ = hcm
pre = torch.sigmoid(mixes[..., :hc_] * scale[0] + base[:hc_]) + 1e-6
print(f'RESULT sigmoid x2     : {bench(lambda: (torch.sigmoid(mixes[..., :hc_]*scale[0]+base[:hc_])+1e-6, torch.sigmoid(mixes[..., hc_:2*hc_]*scale[1]+base[hc_:2*hc_]))):.2f} us')