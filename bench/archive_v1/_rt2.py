import sys, torch, collections
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
p = tok('The capital of France is', return_tensors='pt').input_ids[0].tolist()
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=list(range(40)))
import tinyinfer.deepseek.moe as M
orig = M.Gate.forward
seq = collections.defaultdict(list)
def patched(self, x, image_mask=None):
    w, idx = orig(self, x, image_mask)
    seq[getattr(self,'_lid',-1)].append([int(v) for v in idx.reshape(-1).tolist()])
    return w, idx
M.Gate.forward = patched
for i, b in enumerate(getattr(eng, '_blocks', []) or []):
    m = getattr(b, 'ffn', None)
    if m is not None and hasattr(m, 'gate'): m.gate._lid = i
with torch.inference_mode():
    eng.reset(); list(eng.stream_generate(p, max_new_tokens=20, temperature=0.0))
# 每层的专家频率
print('RESULT 每层不同专家数 / 总访问 / top6 覆盖率')
for lay in sorted(seq)[:6]:
    toks = seq[lay]
    flat = [e for t in toks for e in t]
    cnt = collections.Counter(flat)
    uniq = len(cnt)
    top6 = sum(c for _, c in cnt.most_common(6))
    print(f'  L{lay}: uniq={uniq:3d} n={len(flat):3d} top6={100*top6/max(1,len(flat)):.0f}%')
# 全局：所有层合并的频率分布
allc = collections.Counter()
for lay, toks in seq.items():
    for t in toks:
        for e in t: allc[(lay,e)] += 1
tot = sum(allc.values())
print(f'RESULT 全局: {len(allc)} 个 (layer,expert), 总访问 {tot}')
for frac in (0.1, 0.2, 0.3, 0.5):
    k = max(1, int(len(allc)*frac))
    cov = sum(c for _, c in allc.most_common(k)) / tot
    print(f'  top {frac*100:.0f}% 的专家覆盖 {100*cov:.1f}% 访问')