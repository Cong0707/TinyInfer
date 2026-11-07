import sys, time
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.server import build_backend, GenerationRequest, SamplingParams
b = build_backend("tiny", model_dir=r"E:\Models\Qwen3.5-0.8B", tier="aggressive")
b.warmup()
for temp in (0.0, 1.0):
    req = GenerationRequest(
        request_id="x", messages=[{"role": "user", "content": "hi"}],
        sampling=SamplingParams(max_tokens=64, temperature=temp), model=b.model_id)
    t0 = time.perf_counter(); n = 0; txt = []
    for ev in b.generate(req):
        if ev.token_id is not None:
            n += 1; txt.append(ev.text)
    dt = time.perf_counter() - t0
    print(f"temp={temp}: {n} tok {n/dt:.1f} tok/s  {''.join(txt)[:40]!r}", flush=True)
