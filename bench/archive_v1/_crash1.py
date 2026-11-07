import sys, torch, time
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=list(range(40)))
ids = tok('Answer in one word. The capital of France is', return_tensors='pt').input_ids[0].tolist()
print(f'RESULT prompt len={len(ids)}')
with torch.inference_mode():
    try:
        out = list(eng.stream_generate(ids, max_new_tokens=8, temperature=0.0))
        print(f'RESULT ok: {out}')
    except Exception as e:
        print(f'RESULT FAILED after {len(out) if "out" in dir() else 0}: {repr(e)[:150]}')