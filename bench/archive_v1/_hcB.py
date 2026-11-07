import sys, traceback
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
import torch
from tinyinfer.deepseek import hc
b, s, hcm = 1, 1, 4
mixes = torch.randn(b, s, hcm*hcm + 2*hcm, device='cuda')
scale = torch.rand(3, device='cuda') + 0.5
base = torch.randn(2*hcm + hcm*hcm, device='cuda')
try:
    hc._native_sinkhorn(mixes, scale, base, hcm, 20, 1e-6)
except Exception:
    traceback.print_exc()