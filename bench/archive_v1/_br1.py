import sys, torch, time
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=list(range(40)))
es = eng.expert_store
# 用 CPU perf_counter 在关键位置打点，累计到 dict（单线程视角）
T = {'moe_fetch_wait':0.0,'moe_total':0.0,'attn':0.0,'other':0.0}
import tinyinfer.deepseek.moe as MOE
_of = MOE.MoE.forward
def wf(self, x, *a, **kw):
    t0 = time.perf_counter()
    r = _of(self, x, *a, **kw)
    T['moe_total'] += time.perf_counter()-t0
    return r
MOE.MoE.forward = wf
with torch.inference_mode():
    eng.reset()
    eng.forward(torch.tensor([list(range(100,108))], device='cuda'), 0)
    torch.cuda.synchronize()
    for k in T: T[k]=0.0
    torch.cuda.synchronize(); t0=time.perf_counter()
    eng.forward(torch.tensor([[500]], device='cuda'), start_pos=8)
    torch.cuda.synchronize(); total=time.perf_counter()-t0
print(f'RESULT total={total*1000:.0f}ms moe_total={T["moe_total"]*1000:.0f}ms ({T["moe_total"]/total*100:.0f}%)')
print(f'RESULT 其它(attention+head+hc+python)={(total-T["moe_total"])*1000:.0f}ms')