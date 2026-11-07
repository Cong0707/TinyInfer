import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.kernels import native
def e8m0_to_f(b): return torch.pow(2.0, (b.to(torch.int32) - 127).to(torch.float32))
torch.manual_seed(0)
M, N, K = 128, 128, 128
A = torch.randint(0, 120, (M, K), dtype=torch.uint8, device='cuda')
B = torch.randint(0, 120, (N, K), dtype=torch.uint8, device='cuda')
AS = torch.randint(120, 136, (M, K//32), dtype=torch.uint8, device='cuda')
BS = torch.randint(120, 136, (N//32, K//32), dtype=torch.uint8, device='cuda')
C = torch.empty(M, N, dtype=torch.bfloat16, device='cuda')
native.launch(native.kernel('fp8_gemm_mma','fp8_gemm_mma'), ((N+63)//64, (M+127)//128), (256,),
              [A, AS, B, BS, C, M, N, K])
torch.cuda.synchronize()
Af = A.view(torch.float8_e4m3fn).float(); Bf = B.view(torch.float8_e4m3fn).float()
asf = e8m0_to_f(AS); bsf = e8m0_to_f(BS)
Ct = torch.zeros(M, N, device='cuda', dtype=torch.float32)
for kb in range(K // 32):
    p = Af[:, kb*32:(kb+1)*32] @ Bf[:, kb*32:(kb+1)*32].T
    bsn = bsf[:, kb].repeat_interleave(32, 0)[:N]
    Ct += p * (asf[:, kb][:, None] * bsn[None, :])
# 把 fp32 参考也转成 bf16 再比，隔离 bf16 舍入
Ctb = Ct.to(torch.bfloat16)
exact = (C == Ctb).float().mean().item() * 100
d = (C.float() - Ct).abs()
print(f'RESULT bf16-bit-identical={exact:.2f}%  maxdiff={d.max().item():.4e} relmax={Ct.abs().max().item():.3e}')
print('sample C ', C[0,:4].tolist())
print('sample Ct', Ctb[0,:4].tolist())