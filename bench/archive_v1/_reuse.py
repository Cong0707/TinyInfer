import sys, time
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.server import build_backend, GenerationRequest, SamplingParams
b = build_backend("tiny", model_dir=r"E:\Models\Qwen3.5-0.8B", tier="aggressive")
b.warmup()
q = "1+1等于几？"
for i in range(3):
    req = GenerationRequest(request_id=f"r{i}",
                            messages=[{"role": "user", "content": q}],
                            sampling=SamplingParams(max_tokens=48, temperature=0.0),
                            model=b.model_id)
    t0 = time.perf_counter(); n = 0; txt = []
    events = 0
    for ev in b.generate(req):
        events += 1
        if ev.token_id is not None:
            n += 1; txt.append(ev.text)
    dt = time.perf_counter() - t0
    print(f"call{i}: {n} tok {n/dt:5.1f} tok/s  {''.join(txt)[:50]!r}", flush=True)
