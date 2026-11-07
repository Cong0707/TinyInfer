import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.kernels import cutlass_gemm
def e8m0_to_f(b): return torch.pow(2.0, (b.to(torch.int32) - 127).to(torch.float32))
M, N, K = [int(x) for x in sys.argv[1:4]]
torch.manual_seed(0)
nb = N // 32
A = torch.randint(0, 100, (M, K), dtype=torch.uint8, device='cuda')
B = torch.randint(0, 100, (N, K), dtype=torch.uint8, device='cuda')
AS_f = e8m0_to_f(torch.randint(120,133,(M,K//32),dtype=torch.uint8,device='cuda'))
BS_f = e8m0_to_f(torch.randint(120,133,(nb,K//32),dtype=torch.uint8,device='cuda'))
C = torch.empty(M, N, dtype=torch.bfloat16, device='cuda')
Af = A.view(torch.float8_e4m3fn).float(); Bf = B.view(torch.float8_e4m3fn).float()
Ct = torch.zeros(M, N, device='cuda', dtype=torch.float32)
for kb in range(K // 32):
    p = Af[:, kb*32:(kb+1)*32] @ Bf[:, kb*32:(kb+1)*32].T
    bsn = BS_f[:, kb].repeat_interleave(32, 0)[:N]
    Ct += p * (AS_f[:, kb][:, None] * bsn[None, :])
try:
    cutlass_gemm.fp8_blockwise_gemm(A, AS_f.t().contiguous(), B, BS_f.contiguous(), C)
    torch.cuda.synchronize()
    d = (C.float() - Ct).abs()
    print(f'RESULT {M}x{N}x{K} maxdiff={d.max().item():.4e}')
except Exception as e:
    print(f'RESULT {M}x{N}x{K} FAILED {str(e)[:70]}')