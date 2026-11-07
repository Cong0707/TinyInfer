import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.kernels import native
from tinyinfer.kcheck.more import _e8m0
torch.manual_seed(13)
m, n, k = 128, 64, 64
a = torch.randint(0, 120, (m, k), dtype=torch.uint8, device='cuda')
b = torch.randint(0, 120, (n, k), dtype=torch.uint8, device='cuda')
asf = _e8m0(torch.full((m, k//32), 127, dtype=torch.uint8, device='cuda'))
bsf = _e8m0(torch.full((n//32, k//32), 127, dtype=torch.uint8, device='cuda'))
c = torch.empty(m, n, dtype=torch.bfloat16, device='cuda')
native.launch(native.kernel('fp8_gemm_mma','fp8_gemm_mma'), ((n+63)//64,(m+127)//128), (256,),
              [a, asf, b, bsf, c, m, n, k])
torch.cuda.synchronize()
af = a.view(torch.float8_e4m3fn).float(); bf = b.view(torch.float8_e4m3fn).float()
want = af @ bf.T
d = (c.float()-want).abs()
print('RESULT maxabs', d.max().item(), 'zeros', (c.float()==0).sum().item(), '/', m*n)
rows_ok = [(c[i].float()-want[i]).abs().max().item() < 1e-2 for i in range(m)]
print('RESULT rows correct:', sum(rows_ok), '/', m)
print('RESULT first bad row:', rows_ok.index(False) if not all(rows_ok) else None)
print('RESULT c[0,:4]', c[0,:4].tolist())
print('RESULT w[0,:4]', want[0,:4].tolist())
print('RESULT c[16,:4]', c[16,:4].tolist(), ' w', want[16,:4].tolist())