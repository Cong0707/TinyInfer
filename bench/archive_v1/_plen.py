import sys, torch, time
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=list(range(40)))
tests = [('Hello', 'Hello'),
         ('bench prompt', 'Question: What is the capital of France?\nAnswer:')]
with torch.inference_mode():
    for label, text in tests:
        ids = tok.encode(text, add_special_tokens=False)
        eng.reset()
        list(eng.stream_generate(ids, max_new_tokens=2, temperature=0.0))
        torch.cuda.synchronize()
        ts = []
        for _ in range(4):
            torch.cuda.synchronize(); a=time.perf_counter()
            list(eng.stream_generate(ids, max_new_tokens=1, temperature=0.0))
            torch.cuda.synchronize(); ts.append(time.perf_counter()-a)
        print(f'RESULT {label:14s} prompt_len={len(ids):3d} steps={[round(t*1000) for t in ts]} avg={sum(ts)/len(ts)*1000:.0f}ms')