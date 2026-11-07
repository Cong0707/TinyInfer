import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.deepseek import hc
torch.manual_seed(2)
b, s, hcm = 1, 4, 4
n_scales = 3
mixes = torch.randn(b, s, hcm * hcm + 2 * hcm, device='cuda', dtype=torch.float32)
scale = torch.rand(n_scales, device='cuda', dtype=torch.float32) + 0.5
base = torch.randn(2 * hcm + hcm * hcm, device='cuda', dtype=torch.float32)
eps, iters = 1e-6, 20
ref = hc._sinkhorn_host(mixes, scale, base, hcm, iters, eps)
nat = hc._native_sinkhorn(mixes, scale, base, hcm, iters, eps)
body = hc._sinkhorn_body(mixes, scale, base, hcm, iters, eps)
print('RESULT ref:', [tuple(t.shape) for t in ref])
for label, t in (('native', nat), ('body', body)):
    if t is None:
        print(f'RESULT {label}: None'); continue
    worst = max((a.float()-b2.float()).abs().max().item() for a, b2 in zip(t, ref))
    print(f'RESULT {label}: maxdiff={worst:.3e}')
# 验证 comb 是双随机的
comb = ref[2]
print(f'RESULT comb row sums {comb.sum(-1).flatten()[:4].tolist()}')
print(f'RESULT comb col sums {comb.sum(-2).flatten()[:4].tolist()}')