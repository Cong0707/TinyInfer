import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.kernels import cutlass_gemm

def e8m0_to_f(b): return torch.pow(2.0, (b.to(torch.int32) - 127).to(torch.float32))
def ref_gemm(A, AS_row, B, BS):
    M, K = A.shape; N = B.shape[0]
    Af = A.view(torch.float8_e4m3fn).float(); Bf = B.view(torch.float8_e4m3fn).float()
    asf = e8m0_to_f(AS_row); bsf = e8m0_to_f(BS)
    C = torch.zeros(M, N, device=A.device, dtype=torch.float32)
    for kb in range(K // 32):
        p = Af[:, kb*32:(kb+1)*32] @ Bf[:, kb*32:(kb+1)*32].T
        bsn = bsf[:, kb].repeat_interleave(32, 0)[:N]
        C += p * (asf[:, kb][:, None] * bsn[None, :])
    return C

torch.manual_seed(0)
for (M, N, K) in ((128, 128, 128),):
    nb = N // 32
    A = torch.randint(0, 100, (M, K), dtype=torch.uint8, device='cuda')
    B = torch.randint(0, 100, (N, K), dtype=torch.uint8, device='cuda')
    AS_row = torch.randint(120, 133, (M, K//32), dtype=torch.uint8, device='cuda')
    BS = torch.randint(120, 133, (nb, K//32), dtype=torch.uint8, device='cuda')
    C = torch.empty(M, N, dtype=torch.bfloat16, device='cuda')
    # cutlass wants SFA as [K/32, M] (SFVecSizeM=1 => dim0 is the vector, dim1 stride 1)
    SFA = AS_row.t().contiguous()
    try:
        cutlass_gemm.fp8_blockwise_gemm(A, SFA, B, BS, C)
        torch.cuda.synchronize()
        r = ref_gemm(A, AS_row, B, BS)
        d = (C.float() - r).abs()
        print(f'RESULT M={M} N={N} K={K} maxdiff={d.max().item():.4e} rel={d.max().item()/max(1e-6,r.abs().max().item()):.2e}')
    except Exception as e:
        print(f'RESULT M={M} N={N} K={K} FAILED {str(e)[:150]}')