import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.kernels import native
torch.manual_seed(13)
m, n, k = 16, 64, 64
a = torch.randint(0, 120, (m, k), dtype=torch.uint8, device='cuda')
b = torch.randint(0, 120, (n, k), dtype=torch.uint8, device='cuda')
asf = torch.pow(2.0, (torch.randint(120, 130, (m, k//32), dtype=torch.uint8, device='cuda').int()-127).float())
bsf = torch.pow(2.0, (torch.randint(120, 130, (n//32, k//32), dtype=torch.uint8, device='cuda').int()-127).float())
c = torch.empty(m, n, dtype=torch.bfloat16, device='cuda')
print('RESULT asf dtype', asf.dtype, tuple(asf.shape))
print('RESULT bsf shape', tuple(bsf.shape))
try:
    native.launch(native.kernel('fp8_gemm_mma','fp8_gemm_mma'), ((n+63)//64, (m+127)//128), (256,),
                  [a, asf, b, bsf, c, m, n, k])
    torch.cuda.synchronize()
    print('RESULT c absmax', c.float().abs().max().item(), 'c[:3]', c[0,:3].tolist())
except Exception as e:
    print('RESULT FAIL', repr(e)[:120])
# 参考
af = a.view(torch.float8_e4m3fn).float()
bf = b.view(torch.float8_e4m3fn).float()
want = torch.zeros(m, n, device='cuda')
idx = torch.arange(n, device='cuda') // 32
for kb in range(k//32):
    part = af[:, kb*32:(kb+1)*32] @ bf[:, kb*32:(kb+1)*32].T
    want += part * asf[:, kb][:, None] * bsf[:, kb][idx][None, :]
print('RESULT want absmax', want.abs().max().item())