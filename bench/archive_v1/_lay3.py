import sys, torch, time
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=list(range(40)))
import tinyinfer.deepseek.moe as M
calls = []
orig = M.MoE.forward
def w(self, x, *a, **kw):
    calls.append((self.layer_id, tuple(x.shape), x.size(0) if x.dim()>1 else 1))
    return orig(self, x, *a, **kw)
M.MoE.forward = w
for n in (1, 8):
    ids = list(range(100, 100+n))
    calls.clear()
    with torch.inference_mode():
        eng.reset()
        list(eng.stream_generate(ids, max_new_tokens=2, temperature=0.0))
        calls.clear()
        list(eng.stream_generate(ids, max_new_tokens=1, temperature=0.0))
    rows = [c[2] for c in calls]
    print(f'RESULT len={n}: moe_calls={len(calls)} rows_total={sum(rows)} rows_per_call={rows[:5]}')