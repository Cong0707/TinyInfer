import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
import tinyinfer.kernels.quant as Q
K = 5120
torch.manual_seed(0)
x = torch.randn(1, K, dtype=torch.bfloat16, device='cuda')
a, s = Q.act_quant(x, 32, 'ue8m0', torch.float8_e8m0fnu)
e8 = lambda b: torch.pow(2.0, (b.view(torch.uint8).to(torch.int32)-127).float())
sc = e8(s).repeat_interleave(32, dim=-1)
back = a.float() * sc
print(f'RESULT roundtrip rel err: {((back - x.float()).abs() / x.float().abs().clamp(min=1e-3)).max().item():.4e}')
print(f'RESULT quant range: [{a.float().min().item():.2f}, {a.float().max().item():.2f}] (should be within +-448)')
# 和旧实现对比：临时走慢路径
from tinyinfer.kernels import native
y2 = torch.empty((1,K), dtype=torch.float8_e4m3fn, device='cuda')
s2 = torch.empty(1, K//32, dtype=torch.float8_e8m0fnu, device='cuda')
native.launch(native.kernel('act_quant','act_quant'), ((K//32*32+127)//128,), (128,), [x, y2, s2, 1, K, 1e-4])
torch.cuda.synchronize()
print(f'RESULT fp8 identical to slow path: {torch.equal(a, y2)}')
print(f'RESULT scale identical: {torch.equal(s, s2)}')