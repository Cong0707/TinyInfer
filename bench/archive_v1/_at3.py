import sys, torch, collections
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
p = tok('Hello', return_tensors='pt').input_ids[0].tolist()
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=list(range(40)))
es = eng.expert_store
n_exp = collections.Counter()
of = es.fetch
def wf(layer, experts, **kw):
    n_exp[layer] += len(list(experts))
    return of(layer, experts, **kw)
es.fetch = wf
# 也数 _expert_from_rec 调用（真正的 H2D 点）
import tinyinfer.deepseek.moe as M
orig_rec = M.MoE._expert_from_rec if hasattr(M.MoE,'_expert_from_rec') else None
cnt = collections.Counter()
if orig_rec:
    def wr(self, rec):
        cnt[self.layer_id] += 1
        return orig_rec(self, rec)
    M.MoE._expert_from_rec = wr
with torch.inference_mode():
    eng.reset(); list(eng.stream_generate(p, max_new_tokens=2, temperature=0.0))
    n_exp.clear(); cnt.clear()
    list(eng.stream_generate(p, max_new_tokens=1, temperature=0.0))
    torch.cuda.synchronize()
print('RESULT experts requested:', sum(n_exp.values()))
print('RESULT _expert_from_rec:', sum(cnt.values()))
print('RESULT per-layer rec:', dict(list(cnt.items())[:8]))