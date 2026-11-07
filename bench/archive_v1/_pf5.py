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
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        list(eng.stream_generate(p, max_new_tokens=1, temperature=0.0))
        torch.cuda.synchronize()
    rows = []
    for ev in prof.key_averages():
        rows.append((ev.self_cuda_time_total, ev.self_cpu_time_total, ev.key[:46], ev.count))
    rows.sort(reverse=True)
    print('RESULT top by CUDA time:')
    for c, p_, k, n in rows[:8]:
        print(f'  {k:48s} cuda={c/1000:7.1f}ms cpu={p_/1000:7.1f}ms n={n}')
    rows.sort(key=lambda r: -r[1])
    print('RESULT top by CPU time:')
    for c, p_, k, n in rows[:6]:
        print(f'  {k:48s} cpu={p_/1000:7.1f}ms cuda={c/1000:7.1f}ms n={n}')