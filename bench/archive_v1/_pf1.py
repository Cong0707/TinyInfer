import sys, torch, cProfile, pstats, io
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
    pr = cProfile.Profile(); pr.enable()
    list(eng.stream_generate(p, max_new_tokens=2, temperature=0.0))
    pr.disable()
s = io.StringIO()
pstats.Stats(pr, stream=s).sort_stats('tottime').print_stats(10)
print(s.getvalue()[:2000])