import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.kernels import native

def e8m0_to_f(b): return torch.pow(2.0, (b.to(torch.int32) - 127).to(torch.float32))

def ref_gemm(A, AS, B, BS):
    M, K = A.shape; N = B.shape[0]
    Af = A.view(torch.float8_e4m3fn).float()
    Bf = B.view(torch.float8_e4m3fn).float()
    asf = e8m0_to_f(AS); bsf = e8m0_to_f(BS)
    C = torch.zeros(M, N, device=A.device, dtype=torch.float32)
    for kb in range(K // 32):
        p = Af[:, kb*32:(kb+1)*32] @ Bf[:, kb*32:(kb+1)*32].T
        s = asf[:, kb][:, None] * bsf[:, kb][None, :].repeat_interleave(32, 0)[:N]
        C += p * s
    return C

torch.manual_seed(0)
for (M, N, K) in ((64, 64, 64), (128, 128, 128), (37, 100, 160), (256, 256, 512)):
    A = torch.randint(1, 120, (M, K), dtype=torch.uint8, device='cuda')
    B = torch.randint(1, 120, (N, K), dtype=torch.uint8, device='cuda')
    AS = torch.randint(120, 136, (M, K//32), dtype=torch.uint8, device='cuda')
    BS = torch.randint(120, 136, (N//32 if N % 32 == 0 else N//32+1, K//32), dtype=torch.uint8, device='cuda')
    C = torch.empty(M, N, dtype=torch.bfloat16, device='cuda')
    grid = ((N + 63)//64, (M + 127)//128)
    native.launch(native.kernel('fp8_gemm_mma','fp8_gemm_mma'), grid, (256,),
                  [A, AS, B, BS, C, M, N, K])
    torch.cuda.synchronize()
    r = ref_gemm(A, AS, B, BS)
    d = (C.float() - r).abs()
    print(f'RESULT M={M:4d} N={N:4d} K={K:4d}  maxdiff={d.max().item():.4e}  rel={d.max().item()/max(1e-6, r.abs().max().item()):.2e}')