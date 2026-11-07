import sys, torch, time, os
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=list(range(40)))
es = eng.expert_store
ids = tok('The capital of France is', return_tensors='pt').input_ids[0].tolist()
with torch.inference_mode():
    eng.reset()
    out = list(eng.stream_generate(ids, max_new_tokens=30, temperature=0.0))
    st = es.stats()
n = len(out)
db = st['reader']['bytes_read']; ds = st['reader']['read_seconds']
print(f'RESULT policy={es.ram.policy.name}: n={n} hit={st["ram"]["hit_rate"]:.3f} disk={db/1e9:.1f}GB rate={db/ds/1e9:.2f}GB/s')