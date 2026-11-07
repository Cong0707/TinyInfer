import sys, torch, os
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.deepseek import hc
torch.manual_seed(1)
b, s, hcm, d = 1, 4, 4, 128
x = torch.randn(b, s, d, device='cuda', dtype=torch.bfloat16)
residual = torch.randn(b, s, hcm, d, device='cuda', dtype=torch.bfloat16)
post = torch.randn(b, s, hcm, device='cuda', dtype=torch.float32)
comb = torch.randn(b, s, hcm, hcm, device='cuda', dtype=torch.float32)
ref = hc._hc_post_ref(x, residual, post, comb)
print(f'RESULT ref shape {tuple(ref.shape)}')
# 用 mixin 的方法
import types
obj = types.SimpleNamespace(hc_mult=hcm, layer_id=0, _device='cuda')
for env, label in (({'TI_HC_NOFUSE':'1'}, 'torch/native'), ({}, 'default(fused)')):
    for k,v in env.items(): os.environ[k]=v
    if 'TI_HC_NOFUSE' not in env: os.environ.pop('TI_HC_NOFUSE', None)
    hc._FUSED_POST_OK = None
    try:
        got = hc.HCBlockMixin.hc_post(obj, x, residual, post, comb)
        d_ = (got.float()-ref).abs().max().item()
        print(f'RESULT {label}: maxdiff={d_:.3e} rel={d_/max(1e-6,ref.abs().max().item()):.2e} same_shape={got.shape==ref.shape}')
    except Exception as e:
        print(f'RESULT {label} FAILED {repr(e)[:110]}')