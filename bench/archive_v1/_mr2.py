import sys, torch, time
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.deepseek.fp4_gate_up import gemv_fp4_gate_up
from tinyinfer.deepseek.gemv_fp4_marlin import gemv_fp4_fast
from tinyinfer.deepseek.fp4_native import gemv_fp4_pq, _scale
N, K = 2304, 5120
w1 = torch.randint(0,256,(N,K//2),dtype=torch.uint8,device='cuda')
w3 = torch.randint(0,256,(N,K//2),dtype=torch.uint8,device='cuda')
s1 = torch.randint(120,132,(N,K//32),dtype=torch.uint8,device='cuda')
s3 = torch.randint(120,132,(N,K//32),dtype=torch.uint8,device='cuda')
x = torch.randn(1,K,dtype=torch.bfloat16,device='cuda')
def bench(fn, n=30):
    for _ in range(5): fn()
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t0)/n
try:
    ws1, ws3 = _scale(s1), _scale(s3)
    t1 = bench(lambda: gemv_fp4_gate_up(x, None, w1, ws1, w3, ws3))
    print(f'RESULT gate_up fused : {t1*1e6:6.1f} us')
except Exception as e: print('RESULT gate_up FAIL', repr(e)[:90])
try:
    t2 = bench(lambda: gemv_fp4_fast(x, w1, s1))
    t3 = bench(lambda: (gemv_fp4_fast(x, w1, s1), gemv_fp4_fast(x, w3, s3)))
    print(f'RESULT marlin x1     : {t2*1e6:6.1f} us')
    print(f'RESULT marlin x2     : {t3*1e6:6.1f} us')
except Exception as e: print('RESULT marlin FAIL', repr(e)[:90])