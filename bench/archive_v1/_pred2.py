import sys, torch, time
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=list(range(40)))
b = eng.block(3)
g = b.ffn.gate
x = torch.randn(1, 5120, dtype=torch.bfloat16, device='cuda')
try:
    out = g.forward(x)
    print(f'RESULT gate output shapes: {[tuple(t.shape) for t in out]}')
    print(f'RESULT topk: {out[1].tolist()}')
    def bench(fn,n=500):
        for _ in range(50): fn()
        torch.cuda.synchronize(); t0=time.perf_counter()
        for _ in range(n): fn()
        torch.cuda.synchronize(); return (time.perf_counter()-t0)/n*1e3
    print(f'RESULT gate cost: {bench(lambda: g.forward(x)):.3f} ms')
except Exception as e:
    import traceback; traceback.print_exc()