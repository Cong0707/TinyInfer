import sys, torch, collections, traceback
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
d = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
p = tok("Hello", return_tensors="pt").input_ids[0].tolist()
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device="cuda", layers=list(range(40)))
stats = collections.Counter(); nbytes = collections.Counter()
_orig = torch.Tensor.to
def patched(self, *a, **k):
    try:
        st = traceback.extract_stack(limit=7)
        key = " <- ".join(f"{f.name}" for f in st[-4:-1])
        stats[key] += 1
        nbytes[key] += self.numel() * self.element_size()
    except Exception: pass
    return _orig(self, *a, **k)
torch.Tensor.to = patched
with torch.inference_mode():
    eng.reset(); list(eng.stream_generate(p, max_new_tokens=2, temperature=0.0))
    stats.clear(); nbytes.clear()
    list(eng.stream_generate(p, max_new_tokens=1, temperature=0.0))
torch.Tensor.to = _orig
for k, v in stats.most_common(6):
    print(f"{v:5d}  {nbytes[k]/1e6:8.1f} MB  {k}")
print("total to() calls:", sum(stats.values()))
