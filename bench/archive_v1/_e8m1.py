import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.deepseek.fp4_native import _scale
torch.manual_seed(0)
b = torch.randint(100, 160, (64,), dtype=torch.uint8, device='cuda')
s = b.view(torch.float8_e8m0fnu)
a = s.float()
c = _scale(s)
d = (a.view(torch.int32) != c.view(torch.int32)).sum().item()
print(f'RESULT e8m0->float: diff={d}/{a.numel()}')
print('sample torch', a[:4].tolist())
print('sample mine ', c[:4].tolist())