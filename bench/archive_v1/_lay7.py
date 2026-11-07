import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=list(range(40)))
import tinyinfer.deepseek.moe as M
rows = []
orig = M.MoE.forward
def w(self, x, *a, **kw):
    r = orig(self, x, *a, **kw)
    act = getattr(self, '_last_ids', None)
    if act is not None: rows.append((self.layer_id, len(act), x.size(0) if x.dim()>1 else 1))
    return r
M.MoE.forward = w
for n in (1, 8):
    ids = list(range(100, 100+n))
    rows.clear()
    with torch.inference_mode():
        eng.reset()
        list(eng.stream_generate(ids, max_new_tokens=2, temperature=0.0))
        rows.clear()
        list(eng.stream_generate(ids, max_new_tokens=1, temperature=0.0))
    cnt = [r[1] for r in rows]
    print(f'RESULT len={n}: calls={len(rows)} active_sizes={cnt[:10]} total={sum(cnt)}')