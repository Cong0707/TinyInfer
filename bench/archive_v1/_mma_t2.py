import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.kernels import native
def e8m0_to_f(b): return torch.pow(2.0, (b.to(torch.int32) - 127).to(torch.float32))
torch.manual_seed(0)
M, N, K = 128, 128, 128
# 用 scale=1.0 (e8m0 code 127) 排除 scale 影响
A = torch.full((M, K), 0x38, dtype=torch.uint8, device='cuda')  # 1.0
B = torch.full((N, K), 0x38, dtype=torch.uint8, device='cuda')
AS = torch.full((M, K//32), 127, dtype=torch.uint8, device='cuda')
BS = torch.full((N//32, K//32), 127, dtype=torch.uint8, device='cuda')
C = torch.empty(M, N, dtype=torch.bfloat16, device='cuda')
native.launch(native.kernel('fp8_gemm_mma','fp8_gemm_mma'), ((N+63)//64, (M+127)//128), (256,),
              [A, AS, B, BS, C, M, N, K])
torch.cuda.synchronize()
print('RESULT all-ones K=128 =>', C[0,0].item(), C[64,64].item(), '(want 128)')