import sys, torch, time
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.kernels import mma_gemm
print('RESULT mma_gemm available:', mma_gemm.available() if hasattr(mma_gemm,'available') else '?')
# 测 M=128 的形状
M, N, K = 128, 5120, 5120
try:
    x = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
    w8 = torch.randint(0,120,(N,K),dtype=torch.uint8,device='cuda')
    ws = torch.rand(N//32, K//32, device='cuda')
    out = mma_gemm.fp8_gemm_mma(x, w8, ws)
    print('RESULT mma output:', tuple(out.shape), out.dtype)
except Exception as e:
    print('RESULT mma FAILED:', repr(e)[:150])