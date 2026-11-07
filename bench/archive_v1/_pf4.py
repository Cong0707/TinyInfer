import sys, torch, collections
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=list(range(40)))
visits = []
import tinyinfer.deepseek.moe as M
_o = M.MoE.forward
def w(self, x, *a, **kw):
    r = _o(self, x, *a, **kw)
    ids2 = getattr(self, '_last_ids', None)
    if ids2: visits.append((self.layer_id, tuple(sorted(ids2))))
    return r
M.MoE.forward = w
ids = tok('The quick brown fox jumps over the lazy dog. ' * 2, return_tensors='pt').input_ids[0].tolist()
with torch.inference_mode():
    eng.reset()
    eng.forward(torch.tensor([ids], device='cuda'), 0)
    torch.cuda.synchronize()
print(f'RESULT prompt {len(ids)} tok, MoE calls {len(visits)}')
per_layer = collections.defaultdict(set)
for lid, exps in visits: per_layer[lid].update(exps)
tot = sum(len(v) for v in per_layer.values())
print(f'RESULT distinct (layer,expert): {tot} = {tot*18.8/1000:.1f} GB')
print(f'RESULT visits per layer: {len(visits)//40}')