import sys, torch, collections
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=list(range(40)))
es = eng.expert_store
seq = []
_o = es.ram.get
def w(layer, expert, **kw):
    seq.append((int(layer), int(expert)))
    return _o(layer, expert, **kw)
es.ram.get = w
ids = tok('The capital of France is', return_tensors='pt').input_ids[0].tolist()
with torch.inference_mode():
    eng.reset()
    out = list(eng.stream_generate(ids, max_new_tokens=30, temperature=0.0))
    torch.cuda.synchronize()
c = collections.Counter(seq)
tot = len(seq)
items = c.most_common()
print(f'RESULT total visits {tot}, distinct {len(c)}')
acc = 0
for frac in (0.1, 0.2, 0.3, 0.5, 0.7):
    k = int(len(items)*frac)
    acc = sum(v for _,v in items[:k])
    print(f'RESULT top {frac*100:2.0f}% of combos ({k:4d}): {acc/tot*100:5.1f}% of visits')
# 913 槽能覆盖多少
cover = sum(v for _,v in items[:913])
print(f'RESULT top 913 combos cover: {cover/tot*100:.1f}%')