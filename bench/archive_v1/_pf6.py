import sys, torch
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
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        list(eng.stream_generate(p, max_new_tokens=1, temperature=0.0))
        torch.cuda.synchronize()
    for ev in prof.key_averages():
        dev = getattr(ev, 'self_device_time_total', 0.0)
        cpu = getattr(ev, 'self_cpu_time_total', 0.0)
        if dev > 3000 or cpu > 30000:
            print(f'RESULT {ev.key[:44]:46s} dev={dev/1000:7.1f}ms cpu={cpu/1000:7.1f}ms n={ev.count}')