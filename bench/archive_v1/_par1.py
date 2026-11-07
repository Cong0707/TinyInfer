import sys, torch, time
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=list(range(40)))
es = eng.expert_store
# 记录每层 fetch 的开始/结束，看是否串行
events = []
_o = es._fetch_inner
def w(layer, experts, **kw):
    t0 = time.perf_counter()
    r = _o(layer, experts, **kw)
    events.append((int(layer), t0, time.perf_counter()))
    return r
es._fetch_inner = w
with torch.inference_mode():
    eng.reset()
    eng.forward(torch.tensor([list(range(100,108))], device='cuda'), 0)
    torch.cuda.synchronize()
    events.clear()
    eng.forward(torch.tensor([[500]], device='cuda'), start_pos=8)
    torch.cuda.synchronize()
events.sort(key=lambda e: e[1])
print(f'RESULT {len(events)} layer fetches')
gaps = []
for a, b in zip(events, events[1:]):
    gaps.append((b[1] - a[2]) * 1000)
print(f'RESULT gap between consecutive fetches (ms): {[round(g,1) for g in gaps[:10]]}')
print(f'RESULT mean gap: {sum(gaps)/max(1,len(gaps)):.1f} ms (negative = overlap)')
durs = [(b-a)*1000 for _,a,b in events]
print(f'RESULT fetch durations (ms): {[round(d,1) for d in durs[:8]]}, mean {sum(durs)/len(durs):.1f}')