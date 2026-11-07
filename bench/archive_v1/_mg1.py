import sys, torch, time
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
import tinyinfer.deepseek.moe_gemv as MG
n_exp, k, n = 6, 5120, 2304
aq = torch.randint(0,120,(k,),dtype=torch.uint8,device='cuda')
asc = torch.randint(120,130,(k//32,),dtype=torch.uint8,device='cuda')
w1 = [torch.randint(0,255,(n,k//2),dtype=torch.uint8,device='cuda') for _ in range(n_exp)]
w3 = [torch.randint(0,255,(n,k//2),dtype=torch.uint8,device='cuda') for _ in range(n_exp)]
s1 = [torch.randint(120,130,(n,k//32),dtype=torch.uint8,device='cuda') for _ in range(n_exp)]
s3 = [torch.randint(120,130,(n,k//32),dtype=torch.uint8,device='cuda') for _ in range(n_exp)]
def bench(fn, N=100):
    for _ in range(10): r = fn()
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(N): r = fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t0)/N*1e6
gu = MG.gate_up(aq, asc, w1, s1, w3, s3, n_exp, n, k)
print(f'RESULT gate_up (n_exp=6): {bench(lambda: MG.gate_up(aq, asc, w1, s1, w3, s3, n_exp, n, k)):.1f} us')
we = torch.rand(n_exp, dtype=torch.float32, device='cuda')
print(f'RESULT act           : {bench(lambda: MG.act(gu, we, 0.0)):.1f} us')
print(f'RESULT act_quant     : {bench(lambda: MG.act_quant(gu, we, 0.0)):.1f} us')