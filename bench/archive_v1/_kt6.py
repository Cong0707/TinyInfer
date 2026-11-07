import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.kernels import native
from tinyinfer.kcheck.more import _e8m0
torch.manual_seed(13)
m, n, k = 16, 64, 64
a = torch.randint(0, 120, (m, k), dtype=torch.uint8, device='cuda')
b = torch.randint(0, 120, (n, k), dtype=torch.uint8, device='cuda')
# 用中性 scale 127 -> 1.0
asf = _e8m0(torch.full((m, k//32), 127, dtype=torch.uint8, device='cuda'))
bsf = _e8m0(torch.full((n//32, k//32), 127, dtype=torch.uint8, device='cuda'))
c = torch.empty(m, n, dtype=torch.bfloat16, device='cuda')
native.launch(native.kernel('fp8_gemm_mma','fp8_gemm_mma'), ((n+63)//64,(m+127)//128), (256,),
              [a, asf, b, bsf, c, m, n, k])
torch.cuda.synchronize()
af = a.view(torch.float8_e4m3fn).float(); bf = b.view(torch.float8_e4m3fn).float()
want = (af @ bf.T)
d = (c.float()-want).abs()
print('RESULT neutral scale: maxabs', d.max().item(), 'want max', want.abs().max().item())
print('RESULT zeros:', (c.float()==0).sum().item(), '/', m*n)
# 用大 scale 看溢出
asf2 = _e8m0(torch.full((m, k//32), 130, dtype=torch.uint8, device='cuda'))
print('RESULT scale 130 =', asf2[0,0].item())
c2 = torch.empty(m, n, dtype=torch.bfloat16, device='cuda')
native.launch(native.kernel('fp8_gemm_mma','fp8_gemm_mma'), ((n+63)//64,(m+127)//128), (256,),
              [a, asf2, b, bsf, c, m, n, k])
torch.cuda.synchronize()
print('RESULT scale130 c max', c2.float().abs().max().item(), 'zeros', (c2.float()==0).sum().item())