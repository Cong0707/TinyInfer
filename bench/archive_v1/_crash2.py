import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.server import TinyModelBackend, GenerationRequest, SamplingParams
b = TinyModelBackend(model_dir=r'E:\\models\\DeepSeek-V4.1-Flash-1017', device='cuda', tier='aggressive')
try:
    b.warmup()
    print('RESULT warmup ok')
    req = GenerationRequest(request_id='t', prompt='Answer in one word. The capital of France is',
                            sampling=SamplingParams(max_tokens=8, temperature=0.0))
    evs = list(b._generate(req))
    txt = ''.join((e.text or '') for e in evs)
    print(f'RESULT generated: {txt!r}')
except Exception as e:
    print(f'RESULT FAILED: {repr(e)[:150]}')
finally:
    try: b.close()
    except Exception: pass