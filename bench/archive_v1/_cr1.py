import sys, torch, time
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=list(range(40)))
es = eng.expert_store
with torch.inference_mode():
    eng.reset()
    eng.forward(torch.tensor([list(range(100,108))], device='cuda'), 0)
    torch.cuda.synchronize()
    s0 = es.stats()
    ts = []
    for i in range(6):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        eng.forward(torch.tensor([[500+i]], device='cuda'), start_pos=8+i)
        torch.cuda.synchronize(); ts.append(time.perf_counter()-t0)
    s1 = es.stats()
db = s1['reader']['bytes_read'] - s0['reader']['bytes_read']
ds = s1['reader']['read_seconds'] - s0['reader']['read_seconds']
hit = s1['ram']['hit_rate']
print(f'RESULT 6 步: token_min={min(ts)*1000:.0f}ms avg={sum(ts)/6*1000:.0f}ms')
print(f'RESULT disk={db/1e9:.2f}GB read_wall={ds*1000:.0f}ms ({ds*1000/min(ts)/6*100:.0f}% of per-token)')
print(f'RESULT hit={hit:.3f}')