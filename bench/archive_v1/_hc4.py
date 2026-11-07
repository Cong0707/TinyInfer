import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.deepseek import hc
torch.manual_seed(2)
b, s, hcm = 1, 4, 4
mixes = torch.randn(b, s, hcm * hcm + hcm, device='cuda', dtype=torch.float32)
scale = torch.tensor([1.0], device='cuda')
base = torch.tensor([0.0], device='cuda')
eps = 1e-6
iters = 20
ref = hc._sinkhorn_host(mixes, scale, base, hcm, iters, eps)
nat = hc._native_sinkhorn(mixes, scale, base, hcm, iters, eps)
body = hc._sinkhorn_body(mixes, scale, base, hcm, iters, eps)
for label, t in (('native', nat), ('body', body)):
    if t is None:
        print(f'RESULT {label}: None')
        continue
    d_ = (t.float() - ref.float()).abs().max().item()
    print(f'RESULT {label}: shape={tuple(t.shape)} maxdiff={d_:.3e}')
# 也测 hc_split_sinkhorn
try:
    out = hc.hc_split_sinkhorn(mixes, scale, base, hcm, iters, eps)
    print(f'RESULT hc_split_sinkhorn: {type(out).__name__}', [tuple(x.shape) for x in out] if isinstance(out,(tuple,list)) else '')
except Exception as e:
    print(f'RESULT hc_split_sinkhorn FAILED {repr(e)[:100]}')