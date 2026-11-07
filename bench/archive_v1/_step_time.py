import sys, torch, time
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
    torch.cuda.synchronize()
    # 逐 token 计时
    ts = []
    n = 4
    t0 = time.perf_counter()
    for _ in range(n):
        torch.cuda.synchronize(); a = time.perf_counter()
        list(eng.stream_generate(p, max_new_tokens=1, temperature=0.0))
        torch.cuda.synchronize(); ts.append(time.perf_counter()-a)
print('RESULT per-step ms:', [round(t*1000,1) for t in ts])
print('RESULT avg', round(sum(ts)/len(ts)*1000,1), 'ms')