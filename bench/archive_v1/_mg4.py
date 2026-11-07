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
def bench(fn,N=200):
    for _ in range(10): fn()
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(N): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t0)/N*1e6
gu = MG.gate_up(aq, asc, w1, s1, w3, s3)
we = torch.rand(n_exp, dtype=torch.float32, device='cuda')
print(f'RESULT act      : {bench(lambda: MG.act(gu, we, 0.0)):.1f} us  (was 27.5)')
print(f'RESULT act_quant: {bench(lambda: MG.act_quant(gu, we, 0.0)):.1f} us  (was 34.6)')
# 正确性：和慢路径对比
from tinyinfer.kernels import native
out2 = torch.empty((n_exp, n), dtype=torch.bfloat16, device='cuda')
native.launch(MG._kernel('moe_act6'), ((n+255)//256, n_exp, 1), (256,), [gu, we.contiguous(), out2, n, 0.0, n_exp], 0)
torch.cuda.synchronize()
o1 = MG.act(gu, we, 0.0)
print(f'RESULT act identical: {torch.equal(o1, out2)}')