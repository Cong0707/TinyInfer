import sys, torch, time
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
import tinyinfer.kernels.quant as Q
K = 5120
x = torch.randn(1, K, dtype=torch.bfloat16, device='cuda')
def bench(fn, n=2000):
    for _ in range(50): fn()
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t0)/n*1e6
t = bench(lambda: Q.act_quant(x, 32, 'ue8m0', torch.float8_e8m0fnu))
print(f'RESULT act_quant after bound: {t:.2f} us  (was 26.41)')
# 正确性
a, s = Q.act_quant(x, 32, 'ue8m0', torch.float8_e8m0fnu)
print(f'RESULT shapes {tuple(a.shape)} {tuple(s.shape)} dtype {a.dtype} {s.dtype}')
e8 = lambda b: torch.pow(2.0, (b.view(torch.uint8).to(torch.int32)-127).float())
ref = x.float() / e8(s).repeat_interleave(32, dim=-1)
print(f'RESULT maxdiff vs manual: {(a.float() - ref.clamp(-448,448)).abs().max().item():.4e}')