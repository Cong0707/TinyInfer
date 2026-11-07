import sys, torch, time, os
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=list(range(40)))
es = eng.expert_store; ram = es.ram
# 关掉缓存：get 总是 miss
_orig = ram.get
def noget(layer, expert, **kw):
    kw['admit'] = False
    return _orig(layer, expert, **kw)
ram.get = noget
with torch.inference_mode():
    eng.reset()
    eng.forward(torch.tensor([list(range(100,108))], device='cuda'), 0)
    torch.cuda.synchronize()
    s0 = es.stats()
    ts=[]
    for i in range(4):
        torch.cuda.synchronize(); t0=time.perf_counter()
        eng.forward(torch.tensor([[500+i]], device='cuda'), start_pos=8+i)
        torch.cuda.synchronize(); ts.append(time.perf_counter()-t0)
    s1 = es.stats()
db=(s1['reader']['bytes_read']-s0['reader']['bytes_read'])/1e9
print(f'RESULT no-cache: token={min(ts)*1000:.0f}ms disk={db/4:.2f}GB/tok')
print(f'RESULT 预期(全读): 240 专家 x 18.8MB = {240*18.8/1000:.2f} GB/tok')