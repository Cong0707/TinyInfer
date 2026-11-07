import sys
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.server import build_backend, GenerationRequest, SamplingParams
b = build_backend("tiny", model_dir=r"E:\Models\Qwen3.5-0.8B", tier="aggressive")
b.warmup()
req = GenerationRequest(request_id="t1", messages=[{"role": "user", "content": "hi"}],
                        sampling=SamplingParams(max_tokens=8), model=b.model_id)
n = 0
for ev in b.generate(req):
    n += 1
    print("EV", repr(ev.text), ev.token_id, ev.finish_reason,
          ev.usage.completion_tokens if ev.usage else None, flush=True)
    if n > 20: break
print("total events:", n, flush=True)
