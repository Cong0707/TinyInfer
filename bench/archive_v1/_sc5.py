import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
import tinyinfer.deepseek.gemv_fp4_marlin as M
from tinyinfer.kernels import native
N, K = 2304, 5120
k32 = K//32
s = torch.randint(120,132,(N,k32),dtype=torch.uint8,device='cuda')
e8 = s.view(torch.float8_e8m0fnu)
ref = M._mxfp4_scales(M._permute_scales(e8.T.contiguous(), K, N, 32)).view(torch.uint8)
print(f'RESULT ref {tuple(ref.shape)}')
o = torch.empty(k32, N, dtype=torch.uint8, device='cuda')
b = native.bind('marlin_prep_scales','marlin_prep_scales', ((N*k32+255)//256,), (256,), 'TTii')
b(s, o, N, k32, stream=None)
same = (ref.reshape(-1) == o.reshape(-1)).all().item()
print(f'RESULT exact: {same}')
if not same:
    d = (ref.reshape(-1) != o.reshape(-1)).nonzero().flatten()
    print(f'RESULT mismatches {d.numel()}/{ref.numel()}')
    for i in d[:5].tolist():
        print(f'  idx {i}: ref={ref.reshape(-1)[i].item()} got={o.reshape(-1)[i].item()}')