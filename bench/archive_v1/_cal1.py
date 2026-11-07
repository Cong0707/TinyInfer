import sys, torch, time, collections
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=list(range(40)))
es = eng.expert_store; ram = es.ram
calls = collections.Counter()
_rp = ram._read_many_pooled
def w(blobs, pool=None):
    calls['read_pooled'] += 1
    return _rp(blobs, pool)
ram._read_many_pooled = w
_e8 = ram._read_pinned if hasattr(ram,'_read_pinned') else None
# 统计 fp8_gemv_bs 和 moe_gemv6 的调用
import tinyinfer.kernels.native as NAT
kn = {}
_k = NAT.kernel
def wk(mod, sym):
    f=_k(mod,sym); kn[id(f)]=f'{mod}.{sym}'; return f
NAT.kernel = wk
_o = NAT.launch
def w2(fn, grid, block, args, *a, **kw):
    calls[kn.get(id(fn),'?')] += 1
    return _o(fn, grid, block, args, *a, **kw)
NAT.launch = w2
with torch.inference_mode():
    eng.reset()
    eng.forward(torch.tensor([list(range(100,108))], device='cuda'), 0)
    torch.cuda.synchronize()
    calls.clear()
    torch.cuda.synchronize(); t0=time.perf_counter()
    eng.forward(torch.tensor([[500]], device='cuda'), start_pos=8)
    torch.cuda.synchronize(); total=(time.perf_counter()-t0)*1000
print(f'RESULT total={total:.0f}ms')
for k,v in calls.most_common(10): print(f'RESULT   {k:34s} {v:5d}')