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
print(f'RESULT visits: {len(seq)}')
per_layer = collections.Counter(l for l,_ in seq)
print(f'RESULT visits per layer: min={min(per_layer.values())} max={max(per_layer.values())}')
# 各层的专家分布
layer_experts = collections.defaultdict(set)
for l,e in seq: layer_experts[l].add(e)
sizes = [len(v) for v in layer_experts.values()]
print(f'RESULT distinct experts per layer: min={min(sizes)} max={max(sizes)} mean={sum(sizes)/len(sizes):.0f}')
# 关键：跨层是否有重叠（如果有，分层缓存会浪费）
all_e = collections.Counter(e for _,e in seq)
print(f'RESULT total distinct (layer,expert): {len(set(seq))}')
print(f'RESULT distinct expert ids only: {len(all_e)}')