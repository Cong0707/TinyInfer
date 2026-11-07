import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.kernels.fp4q import fp4_act_quant as mine
from tinyinfer.official_kernel_ref import fp4_act_quant as ref
torch.manual_seed(0)
shape, block, sd = (1,512), 32, torch.float8_e8m0fnu
x = torch.randn(*shape, dtype=torch.bfloat16, device='cuda')
a = x.clone(); b = x.clone()
ra = ref(a, block, True, sd)
rb = mine(b, block, True, sd)
d = (ra.view(torch.uint8) != rb.view(torch.uint8)).flatten()
print('diff', d.sum().item(), 'of', d.numel())
idx = d.nonzero().flatten()[:5].tolist()
for i in idx:
    print(f'  i={i} x={x.flatten()[i].item():.6f} ref={ra.flatten()[i].item():.6f} mine={rb.flatten()[i].item():.6f}')
if idx:
    g = idx[0]//32
    print('group', g, 'absmax', x.flatten()[g*32:(g+1)*32].abs().max().item())