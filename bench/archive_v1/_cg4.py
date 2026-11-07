import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.kernels import cutlass_gemm
def e8m0_to_f(b): return torch.pow(2.0, (b.to(torch.int32) - 127).to(torch.float32))
torch.manual_seed(0)
# 只测对齐的、大一点的形状，判断是不是所有形状都崩
for (M, N, K) in ((128,128,128), (256,128,128), (512,256,256)):
    nb = N // 32
    A = torch.randint(0, 100, (M, K), dtype=torch.uint8, device='cuda')
    B = torch.randint(0, 100, (N, K), dtype=torch.uint8, device='cuda')
    AS_f = e8m0_to_f(torch.randint(120,133,(M,K//32),dtype=torch.uint8,device='cuda'))
    BS_f = e8m0_to_f(torch.randint(120,133,(nb,K//32),dtype=torch.uint8,device='cuda'))
    C = torch.empty(M, N, dtype=torch.bfloat16, device='cuda')
    try:
        cutlass_gemm.fp8_blockwise_gemm(A, AS_f.t().contiguous(), B, BS_f.contiguous(), C)
        torch.cuda.synchronize()
        print(f'RESULT M={M} N={N} K={K} OK  C[0,0]={C[0,0].item():.3f}')
    except Exception as e:
        print(f'RESULT M={M} N={N} K={K} FAILED {str(e)[:80]}')