import sys, torch, json
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
p = tok('Hello', return_tensors='pt').input_ids[0].tolist()
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=list(range(40)))
with torch.inference_mode():
    eng.reset(); list(eng.stream_generate(p, max_new_tokens=2, temperature=0.0))
    s0 = eng.expert_store.stats()
    list(eng.stream_generate(p, max_new_tokens=4, temperature=0.0))
    s1 = eng.expert_store.stats()
    dr = s1['reader']['read_seconds'] - s0['reader']['read_seconds']
    dg = s1['reader']['bytes_read'] - s0['reader']['bytes_read']
    dm = s1['ram']['misses'] - s0['ram']['misses']
    print(f'RESULT 4 tokens: disk_reads={dm} bytes={dg/2**30:.2f}GB read_time={dr*1000:.0f}ms')
    print(f'RESULT 每token disk={dr/4*1000:.0f}ms')