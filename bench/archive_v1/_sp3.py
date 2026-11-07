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
    eng.reset(); list(eng.stream_generate(p, max_new_tokens=3, temperature=0.0))
    torch.cuda.synchronize()
    e0 = torch.cuda.Event(True); e1 = torch.cuda.Event(True)
    t0 = time.perf_counter(); e0.record()
    list(eng.stream_generate(p, max_new_tokens=3, temperature=0.0))
    e1.record(); torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    gpu = e0.elapsed_time(e1)/1000.0
print(f'RESULT wall={wall/3*1000:.0f}ms gpu={gpu/3*1000:.0f}ms wait={(wall-gpu)/3*1000:.0f}ms')