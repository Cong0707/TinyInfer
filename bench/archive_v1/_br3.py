import sys, torch, time
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=list(range(40)))
es = eng.expert_store; ram = es.ram
T = {'fetch':0.0,'gate':0.0,'shared':0.0,'serve':0.0}
import tinyinfer.deepseek.moe as M
_g = M.Gate.forward
def wg(self, x, *a):
    t0=time.perf_counter(); r=_g(self,x,*a); T['gate']+=time.perf_counter()-t0; return r
M.Gate.forward = wg
_f = es.fetch
def wf(layer, experts, **kw):
    t0=time.perf_counter(); r=_f(layer, experts, **kw); T['fetch']+=time.perf_counter()-t0; return r
es.fetch = wf
with torch.inference_mode():
    eng.reset()
    eng.forward(torch.tensor([list(range(100,108))], device='cuda'), 0)
    torch.cuda.synchronize()
    for k in T: T[k]=0.0
    torch.cuda.synchronize(); t0=time.perf_counter()
    eng.forward(torch.tensor([[500]], device='cuda'), start_pos=8)
    torch.cuda.synchronize(); total=(time.perf_counter()-t0)*1000
print(f'RESULT total={total:.0f}ms')
for k,v in T.items(): print(f'RESULT   {k:8s} {v*1000:6.0f}ms {v*1000/total*100:4.0f}%')
print(f'RESULT   rest     {total-sum(T.values())*1000:6.0f}ms')