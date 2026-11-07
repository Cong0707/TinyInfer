import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.server import TinyModelBackend, GenerationRequest, SamplingParams
req = GenerationRequest(request_id='t1',
                        messages=[{'role':'user','content':'The capital of France is'}],
                        sampling=SamplingParams(max_tokens=5, temperature=0.0))
for tier in ('conservative', 'aggressive'):
    b = TinyModelBackend(model_dir=r'E:\\models\\DeepSeek-V4.1-Flash-1017',
                         device='cuda', tier=tier)
    try:
        b.load()
        text = b.render_prompt(req)
        evs = list(b._generate(req))
        txt = ''.join((e.text or '') for e in evs)
        print(f'RESULT tier={tier}: {txt!r}')
    finally:
        try: b.close()
        except Exception: pass
        import gc; gc.collect(); torch.cuda.empty_cache()