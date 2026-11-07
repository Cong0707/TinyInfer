import sys, torch, collections, json
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
p = tok('The capital of France is', return_tensors='pt').input_ids[0].tolist()
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=list(range(40)))
# 抓每层每 token 的路由
import tinyinfer.deepseek.moe as M
orig = M.Gate.forward
seq = collections.defaultdict(list)   # layer -> [set of experts per token]
lid = {'cur': 0}
def patched(self, x, image_mask=None):
    r = orig(self, x, image_mask)
    w, idx = r
    lay = getattr(self, '_lid', None)
    if lay is None:
        # 用 weight 指针反查层号
        lay = hash(int(self.weight.data_ptr())) % 10000
    seq[lay].append(set(int(v) for v in idx.reshape(-1).tolist()))
    return r
M.Gate.forward = patched
# 给每层 gate 打上 lid
for i, b in enumerate(getattr(eng, '_blocks', []) or []):
    m = getattr(b, 'ffn', None)
    if m is not None and hasattr(m, 'gate'):
        m.gate._lid = i
with torch.inference_mode():
    eng.reset(); list(eng.stream_generate(p, max_new_tokens=10, temperature=0.0))
tot = sum(len(v) for v in seq.values())
print('RESULT layers captured:', len(seq), 'total records:', tot)
# 相邻 token 的重叠率
ov = []
for lay, toks in seq.items():
    for a, b2 in zip(toks, toks[1:]):
        if a or b2: ov.append(len(a & b2) / max(1, len(a | b2)))
print(f'RESULT adjacent-token Jaccard: mean={sum(ov)/max(1,len(ov)):.3f} n={len(ov)}')
# 基线：随机选 6 个的期望 Jaccard
import random
rnd = []
for lay, toks in seq.items():
    for a in toks:
        b2 = set(random.sample(range(384), 6))
        if a or b2: rnd.append(len(a & b2) / max(1, len(a | b2)))
print(f'RESULT random baseline: {sum(rnd)/max(1,len(rnd)):.3f}')