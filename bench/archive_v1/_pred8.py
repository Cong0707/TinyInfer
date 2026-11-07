import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=list(range(40)))
eng.enable_prefetch = True
preds = {}
_o = eng._prefetch_layer
def w(layer_id, experts):
    preds[layer_id] = list(experts)
    return _o(layer_id, experts)
eng._prefetch_layer = w
with torch.inference_mode():
    eng.reset()
    eng.forward(torch.tensor([list(range(100,108))], device='cuda'), 0)
    torch.cuda.synchronize()
    preds.clear()
    eng.forward(torch.tensor([[500]], device='cuda'), start_pos=8)
    torch.cuda.synchronize()
m = t = 0
for lid, exps in preds.items():
    real = eng._last_routing.get(lid)
    if real:
        t += 1
        if set(exps) == set(real): m += 1
print(f'RESULT prefetch calls={len(preds)} exact_match={m}/{t}')
if preds: print(f'RESULT sizes: {sorted(set(len(v) for v in preds.values()))}')