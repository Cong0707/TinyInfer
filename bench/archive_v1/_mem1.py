# 每 token 实际传了多少
import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
p = tok('Hello', return_tensors='pt').input_ids[0].tolist()
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=list(range(40)))
from torch.profiler import profile, ProfilerActivity
with torch.inference_mode():
    eng.reset(); list(eng.stream_generate(p, max_new_tokens=2, temperature=0.0))
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        list(eng.stream_generate(p, max_new_tokens=1, temperature=0.0))
        torch.cuda.synchronize()
    # 统计每个 memcpy 的大小分布
    evs = list(prof.events())
    sizes = []
    for ev in evs:
        if 'HtoD' in str(ev.key) or 'Memcpy' in str(ev.key):
            sizes.append(ev.device_time_total)
    print(f'RESULT memcpy count={len(sizes)}')