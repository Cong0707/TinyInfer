import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
p = tok('Hello', return_tensors='pt').input_ids[0].tolist()
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=list(range(40)))
print('RESULT coalesced layers:', len(getattr(eng, '_flat_coalesced', {})))
from torch.profiler import profile, ProfilerActivity
with torch.inference_mode():
    eng.reset(); list(eng.stream_generate(p, max_new_tokens=2, temperature=0.0))
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        list(eng.stream_generate(p, max_new_tokens=1, temperature=0.0))
        torch.cuda.synchronize()
    n = sum(1 for ev in prof.events() if 'Memcpy' in str(ev.key))
    h = [ev for ev in prof.key_averages() if 'HtoD' in str(ev.key)]
    for ev in h:
        print(f'RESULT {ev.key[:30]:32s} dev={ev.self_device_time_total/1000:7.1f}ms n={ev.count}')
    print(f'RESULT total memcpy events={n}')