import sys, torch
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.server import TinyModelBackend, GenerationRequest, SamplingParams
import inspect
req = GenerationRequest(messages=[{'role':'user','content':'The capital of France is'}],
                        sampling=SamplingParams(max_tokens=5, temperature=0.0))
b = TinyModelBackend(model_dir=r'E:\\models\\DeepSeek-V4.1-Flash-1017', backend='tiny',
                     device='cuda', tier='aggressive', adapter='auto')
try:
    b.load()
    text = b.render_prompt(req)
    ids = b._encode(text)
    print('RESULT rendered:', repr(text))
    print('RESULT ids:', ids)
    print('RESULT decode back:', repr(b._decode(ids)))
    evs = list(b._generate(req))
    txt = ''.join(e.text or '' for e in evs)
    print('RESULT generated:', repr(txt))
finally:
    try: b.close()
    except Exception: pass