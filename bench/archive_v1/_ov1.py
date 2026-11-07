import sys, torch, time
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=list(range(40)))
es = eng.expert_store; ram = es.ram
times = []
_o = ram._read_many_pooled
def w(blobs, pool=None):
    t0 = time.perf_counter()
    r = _o(blobs, pool)
    times.append((len(blobs), (time.perf_counter()-t0)*1000))
    return r
ram._read_many_pooled = w
with torch.inference_mode():
    eng.reset()
    eng.forward(torch.tensor([list(range(100,108))], device='cuda'), 0)
    torch.cuda.synchronize()
    times.clear()
    t0=time.perf_counter()
    eng.forward(torch.tensor([[500]], device='cuda'), start_pos=8)
    torch.cuda.synchronize()
    total=(time.perf_counter()-t0)*1000
sum_read = sum(t for _,t in times)
print(f'RESULT step total: {total:.0f} ms')
print(f'RESULT read calls: {len(times)}, sum of call durations: {sum_read:.0f} ms ({sum_read/total*100:.0f}% of step)')
print(f'RESULT per call ms: {[round(t) for _,t in times[:8]]}')
print(f'RESULT blob counts: {[n for n,_ in times[:8]]}')