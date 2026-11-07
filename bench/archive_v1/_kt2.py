import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.kcheck.more import _ref_w4a8, _e8m0
torch.manual_seed(11)
n, k = 256, 512
w = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device='cuda')
ws = torch.rand(n, k // 32, device='cuda') + 0.5
aq = torch.randint(0, 120, (k,), dtype=torch.uint8, device='cuda')
asc = (torch.rand(k // 32, device='cuda') * 0.2 + 0.01).to(torch.float32)
try:
    r = _ref_w4a8(aq, asc, w, ws)
    print('RESULT ref[:3]', r[:3].tolist(), 'absmax', r.abs().max().item())
except Exception as e:
    import traceback; traceback.print_exc()