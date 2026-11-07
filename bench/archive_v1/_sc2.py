import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
import tinyinfer.deepseek.gemv_fp4_marlin as M
from tinyinfer.kernels import native
N, K = 2304, 5120
n32, k32 = N//32, K//32
s = torch.randint(120,132,(N,K//32),dtype=torch.uint8,device='cuda')
e8 = s.reshape(N, k32).view(torch.float8_e8m0fnu)
ref = M._mxfp4_scales(M._permute_scales(e8.T.contiguous(), K, N, 32)).view(torch.uint8)
# 我的 kernel
o = torch.empty(k32, n32, dtype=torch.uint8, device='cuda')
b = native.bind('marlin_prep_scales','marlin_prep_scales', ((n32*k32+255)//256,), (256,), 'TTii')
b(s.view(torch.uint8), o, n32, k32, stream=None)
same = (ref == o).all().item()
print(f'RESULT exact match: {same}')
if not same:
    diff = (ref != o)
    print(f'RESULT mismatches: {diff.sum().item()} / {ref.numel()}')
    idx = diff.nonzero()[:3]
    for i in idx:
        r_, c_ = i.tolist()
        print(f'  [{r_},{c_}] ref={ref[r_,c_].item()} got={o[r_,c_].item()}')