import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.deepseek import gemv_fp4_marlin as M
torch.cuda.init(); torch.zeros(1, device='cuda')
N, K = 2304, 5120
w = torch.randint(0,256,(N,K//2),dtype=torch.uint8,device='cuda')
s = torch.randint(120,132,(N,K//32),dtype=torch.uint8,device='cuda')
x = torch.randn(1,K,dtype=torch.bfloat16,device='cuda')
# bf16 版能跑通
out = M.gemv_fp4_fast(x, w, s)
torch.cuda.synchronize()
print(f'RESULT bf16 ok {tuple(out.shape)}')
# 看 bf16 的调用参数
wt = M._prepared(w, s, None)
launch, smem = M._native_launcher()
print(f'RESULT bf16 args: a={tuple(x.shape)} qw={tuple(wt.qweight.shape)} sc={tuple(wt.scales.shape)}')
print(f'RESULT workspace {tuple(M._workspace(x.device).shape)}')