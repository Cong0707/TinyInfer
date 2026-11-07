import sys, torch, time
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
import tinyinfer.deepseek.moe_gemv as MG
n_exp, k, n = 6, 5120, 2304
aq = torch.randint(0,120,(k,),dtype=torch.uint8,device='cuda')
asc = torch.randint(120,130,(k//32,),dtype=torch.uint8,device='cuda')
mk = lambda sh: [torch.randint(0,255,sh,dtype=torch.uint8,device='cuda') for _ in range(n_exp)]
w1=mk((n,k//2)); w3=mk((n,k//2))
s1=[torch.randint(120,130,(n,k//32),dtype=torch.uint8,device='cuda') for _ in range(n_exp)]
s3=[torch.randint(120,130,(n,k//32),dtype=torch.uint8,device='cuda') for _ in range(n_exp)]
def bench(f,N2=100):
    for _ in range(10): f()
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(N2): f()
    torch.cuda.synchronize(); return (time.perf_counter()-t0)/N2*1e6
t = bench(lambda: MG.gate_up(aq, asc, w1, s1, w3, s3))
print(f'RESULT gate_up (全部6专家, 1次launch): {t:.1f} us')
print(f'RESULT 若每专家一次: 6 x 16.5 = 99 us 的 driver 成本, 再加 6 倍计算')
# 计算量：6 专家 x 2 矩阵 x 2304 x 5120 半字节
bytes_total = 6*2*(n*k//2 + n*(k//32))
print(f'RESULT 权重总量: {bytes_total/2**20:.1f} MB, 带宽 {bytes_total/t/1e9*1e6/1e6:.0f} GB/s')