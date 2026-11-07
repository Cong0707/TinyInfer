import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=list(range(40)))
eng.enable_prefetch = True
hits = {'pred':0, 'real':0, 'match':0}
preds = {}
_orig = eng._prefetch_layer
def wp(layer_id, experts):
    hits['pred'] += 1
    preds[layer_id] = list(experts)
    return _orig(layer_id, experts)
eng._prefetch_layer = wp
with torch.inference_mode():
    eng.reset()
    eng.forward(torch.tensor([list(range(100,108))], device='cuda'), 0)
    torch.cuda.synchronize()
    preds.clear(); hits['pred']=0
    eng.forward(torch.tensor([[500]], device='cuda'), start_pos=8)
    torch.cuda.synchronize()
m = 0; t = 0
for lid, exps in preds.items():
    real = eng._last_routing.get(lid)
    if real:
        t += 1
        if set(exps) == set(real): m += 1
print(f'RESULT prefetch calls: {hits["pred"]}')
print(f'RESULT exact match with real routing: {m}/{t}')
print(f'RESULT sample pred: {list(preds.items())[:2]}')