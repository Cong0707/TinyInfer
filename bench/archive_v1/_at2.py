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
calls = collections.Counter()
of = es.fetch
def wf(layer, experts, **kw):
    calls[layer] += 1
    return of(layer, experts, **kw)
es.fetch = wf
with torch.inference_mode():
    eng.reset(); list(eng.stream_generate(p, max_new_tokens=2, temperature=0.0))
    calls.clear()
    list(eng.stream_generate(p, max_new_tokens=1, temperature=0.0))
    torch.cuda.synchronize()
print('RESULT fetch per layer:', dict(list(calls.items())[:8]))
print('RESULT total layers:', len(calls), 'total calls:', sum(calls.values()))