import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.kernels import native
from tinyinfer.kcheck.more import _e8m0
torch.manual_seed(13)
m, n, k = 16, 64, 64
a = torch.randint(0, 120, (m, k), dtype=torch.uint8, device='cuda')
b = torch.randint(0, 120, (n, k), dtype=torch.uint8, device='cuda')
asf = _e8m0(torch.randint(120, 130, (m, k//32), dtype=torch.uint8, device='cuda'))
bsf = _e8m0(torch.randint(120, 130, (n//32, k//32), dtype=torch.uint8, device='cuda'))
c = torch.empty(m, n, dtype=torch.bfloat16, device='cuda')
native.launch(native.kernel('fp8_gemm_mma','fp8_gemm_mma'), ((n+63)//64,(m+127)//128), (256,),
              [a, asf, b, bsf, c, m, n, k])
torch.cuda.synchronize()
af = a.view(torch.float8_e4m3fn).float(); bf = b.view(torch.float8_e4m3fn).float()
want = torch.zeros(m, n, device='cuda')
idx = torch.arange(n, device='cuda') // 32
for kb in range(k//32):
    part = af[:, kb*32:(kb+1)*32] @ bf[:, kb*32:(kb+1)*32].T
    want += part * asf[:, kb][:, None] * bsf[idx, kb][None, :]
d = (c.float()-want).abs()
print('RESULT maxabs', d.max().item(), 'max want', want.abs().max().item())
i = d.argmax().item(); r_, c_ = i//n, i%n
print(f'RESULT worst at [{r_},{c_}] got={c[r_,c_].item()} want={want[r_,c_].item()}')
print('RESULT zero outputs:', (c.float()==0).sum().item(), '/', m*n)
print('RESULT relative error overall:', (d.max()/want.abs().max()).item())