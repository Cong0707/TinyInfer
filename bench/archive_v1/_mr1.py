import sys, torch, time
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.deepseek.gemv_fp4_marlin import gemv_fp4_fast
N, K = 2304, 5120
w = torch.randint(0,256,(N,K//2),dtype=torch.uint8,device='cuda')
s = torch.randint(120,132,(N,K//32),dtype=torch.uint8,device='cuda')
x = torch.randn(1,K,dtype=torch.bfloat16,device='cuda')
try:
    out = gemv_fp4_fast(x, w, s)
    print('RESULT out:', tuple(out.shape))
    for _ in range(5): gemv_fp4_fast(x,w,s)
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(50): gemv_fp4_fast(x,w,s)
    torch.cuda.synchronize(); dt=(time.perf_counter()-t0)/50
    b = N*K//2 + N*K//32 + N*2
    print(f'RESULT marlin: {dt*1e6:.1f} us  {b/dt/1e9:.1f} GB/s')
except Exception as e:
    print('RESULT FAILED:', repr(e)[:180])