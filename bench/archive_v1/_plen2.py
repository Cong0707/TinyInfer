import sys, torch, time
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=list(range(40)))
with torch.inference_mode():
    for n in (1, 2, 4, 8, 11):
        ids = list(range(100, 100+n))
        eng.reset()
        list(eng.stream_generate(ids, max_new_tokens=2, temperature=0.0))
        torch.cuda.synchronize()
        ts = []
        for _ in range(3):
            torch.cuda.synchronize(); a=time.perf_counter()
            list(eng.stream_generate(ids, max_new_tokens=1, temperature=0.0))
            torch.cuda.synchronize(); ts.append(time.perf_counter()-a)
        print(f'RESULT prompt_len={n:3d} avg={sum(ts)/len(ts)*1000:.0f}ms')