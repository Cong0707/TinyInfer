import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.deepseek import hc
print('RESULT native_ok:', hc._native_ok())
print('RESULT has hc_pre_rms_norm:', hasattr(hc, 'hc_pre_rms_norm'))
torch.manual_seed(0)
b, s, hcm, d = 1, 4, 4, 128
x = torch.randn(b, s, hcm, d, device='cuda', dtype=torch.bfloat16)
pre_mix = torch.randn(b, s, hcm, device='cuda', dtype=torch.float32)
ref = hc._hc_pre_ref(x, pre_mix)
print(f'RESULT ref shape {tuple(ref.shape)}')
try:
    got = hc.hc_pre_rms_norm(x, pre_mix, rms_eps=1e-6)
    print(f'RESULT native shape {tuple(got.shape)}')
except Exception as e:
    import traceback; traceback.print_exc()