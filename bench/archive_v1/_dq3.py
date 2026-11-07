import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.quant import dequant_fp8_block, e8m0_to_float
from tinyinfer.dequant_kernel import dequant_fp8
torch.manual_seed(0)
for (N,K) in ((512,512), (2304,5120)):
    w = torch.randn(N, K, dtype=torch.float16, device='cuda').to(torch.float8_e4m3fn)
    s = torch.ones(N//32, K//32, dtype=torch.float8_e8m0fnu, device='cuda')
    a = dequant_fp8(w, s)
    bo, bi = N//32, K//32
    sf = e8m0_to_float(s)[:bo,:bi]
    w4 = w.to(torch.float32).view(bo,32,bi,32)
    b = (w4 * sf[:,None,:,None]).reshape(N,K)
    print(f'RESULT N={N} K={K}: kernel dtype={a.dtype} torch dtype={b.dtype} maxdiff={(a.float()-b.float()).abs().max().item():.3e}')