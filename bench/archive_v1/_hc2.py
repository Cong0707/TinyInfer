import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.deepseek import hc
import torch.nn.functional as F
torch.manual_seed(0)
b, s, hcm, d = 1, 4, 4, 128
x = torch.randn(b, s, hcm, d, device='cuda', dtype=torch.bfloat16)
pre_mix = torch.randn(b, s, hcm, device='cuda', dtype=torch.float32)
w = torch.randn(d, device='cuda', dtype=torch.bfloat16)
eps = 1e-6
# 参考：先 collapse 再 rmsnorm
collapsed = hc._hc_pre_ref(x, pre_mix)
ref = F.rms_norm(collapsed.float(), (d,), w.float(), eps)
import os
for fuse in ('1','0'):
    os.environ['TI_FUSE'] = fuse
    hc._FUSED_PRE_NORM = None
    got = hc.hc_pre_rms_norm(x, pre_mix, w, eps)
    d_ = (got.float()-ref).abs().max().item()
    print(f'RESULT TI_FUSE={fuse}: shape={tuple(got.shape)} maxdiff={d_:.3e} rel={d_/max(1e-6,ref.abs().max().item()):.2e}')