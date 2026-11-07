import sys, torch, time
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=list(range(40)))
ids = list(range(100, 108))  # 8 tokens
with torch.inference_mode():
    eng.reset()
    list(eng.stream_generate(ids, max_new_tokens=2, temperature=0.0))
    torch.cuda.synchronize()
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        list(eng.stream_generate(ids, max_new_tokens=1, temperature=0.0))
        torch.cuda.synchronize()
    rows = []
    for ev in prof.key_averages():
        d_ = getattr(ev, 'self_device_time_total', 0.0)
        c_ = getattr(ev, 'self_cpu_time_total', 0.0)
        if d_ > 2000 or c_ > 5000:
            rows.append((max(d_, c_), d_, c_, ev.key[:44], ev.count))
    rows.sort(reverse=True)
    for _, d_, c_, k, n in rows[:12]:
        print(f'RESULT {k:46s} dev={d_/1000:7.1f}ms cpu={c_/1000:7.1f}ms n={n}')