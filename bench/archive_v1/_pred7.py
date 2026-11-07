import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=list(range(40)))
print(f'RESULT enable_prefetch default: {eng.enable_prefetch}')
print(f'RESULT predictor: {eng.predictor}')
# 先看 _last_routing 与真实路由的一致性（跨 token）
with torch.inference_mode():
    eng.reset()
    eng.forward(torch.tensor([list(range(100,108))], device='cuda'), 0)
    torch.cuda.synchronize()
    prev = dict(eng._last_routing)
    for i in range(3):
        eng.forward(torch.tensor([[500+i]], device='cuda'), start_pos=8+i)
    torch.cuda.synchronize()
    cur = dict(eng._last_routing)
same = sum(1 for k in cur if k in prev and set(cur[k])==set(prev[k]))
jacc = []
for k in cur:
    if k in prev:
        a,b = set(cur[k]), set(prev[k])
        if a|b: jacc.append(len(a&b)/len(a|b))
print(f'RESULT layers={len(cur)}, exact same as prev token: {same}')
print(f'RESULT mean Jaccard vs prev token: {sum(jacc)/max(1,len(jacc)):.3f}')