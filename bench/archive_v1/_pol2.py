import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=list(range(40)))
es = eng.expert_store
print(f'RESULT initial ram entries: {es.stats()["ram"]["entries"]}')
with torch.inference_mode():
    eng.reset()
    eng.forward(torch.tensor([list(range(100,108))], device='cuda'), 0)
    torch.cuda.synchronize()
    s0 = es.stats()
    print(f'RESULT after prefill: entries={s0["ram"]["entries"]} hit={s0["ram"]["hit_rate"]:.3f}')
    for i in range(12):
        eng.forward(torch.tensor([[500+i]], device='cuda'), start_pos=8+i)
    torch.cuda.synchronize()
    s1 = es.stats()
    b0 = s0['reader']['bytes_read']; b1 = s1['reader']['bytes_read']
print(f'RESULT after 12 decode: entries={s1["ram"]["entries"]} hit={s1["ram"]["hit_rate"]:.3f}')
print(f'RESULT disk per token: {(b1-b0)/12/1e9:.2f} GB')