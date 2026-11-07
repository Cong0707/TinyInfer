import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=list(range(40)))
import tinyinfer.deepseek.moe as M
recs = []
orig = M.Gate.forward
def w(self, x, image_mask=None):
    r = orig(self, x, image_mask)
    recs.append((x.shape, r[0].shape, r[1].shape))
    return r
M.Gate.forward = w
for n in (1, 8):
    ids = list(range(100, 100+n))
    recs.clear()
    with torch.inference_mode():
        eng.reset()
        list(eng.stream_generate(ids, max_new_tokens=2, temperature=0.0))
        recs.clear()
        list(eng.stream_generate(ids, max_new_tokens=1, temperature=0.0))
    print(f'RESULT len={n}: gate calls={len(recs)}')
    for r in recs[:4]: print('   ', r)