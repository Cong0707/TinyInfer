import sys, torch
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
    for i in range(8):
        eng.forward(torch.tensor([[500+i]], device='cuda'), start_pos=8+i)
    torch.cuda.synchronize()
    s1 = es.stats()
db = s1['reader']['bytes_read']-s0['reader']['bytes_read']
print(f'RESULT 8 decode steps: {db/1e9:.2f} GB total => {db/8/1e9:.2f} GB/token')
r0 = s0['ram']; r1 = s1['ram']
print(f'RESULT ram entries: {r0["entries"]} -> {r1["entries"]}')
print(f'RESULT ram hit_rate: {r0["hit_rate"]:.3f} -> {r1["hit_rate"]:.3f}')
print(f'RESULT reader gbps: {s1["reader"]["gbps"]:.2f}')