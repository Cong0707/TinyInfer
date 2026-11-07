import sys, torch, time
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=list(range(40)))
text = 'The quick brown fox jumps over the lazy dog. ' * 5
ids = tok(text, return_tensors='pt').input_ids[0].tolist()
print(f'RESULT prompt len={len(ids)}')
with torch.inference_mode():
    eng.reset()
    t0=time.perf_counter()
    out = list(eng.stream_generate(ids, max_new_tokens=4, temperature=0.0))
    torch.cuda.synchronize()
    dt=time.perf_counter()-t0
print(f'RESULT prefill+4decode: {dt*1000:.0f}ms out={out}')
print(f'RESULT decoded: {tok.decode(out)!r}')
# 再跑一次验证可重复
with torch.inference_mode():
    eng.reset()
    torch.cuda.synchronize(); t0=time.perf_counter()
    out2 = list(eng.stream_generate(ids, max_new_tokens=4, temperature=0.0))
    torch.cuda.synchronize()
    print(f'RESULT second run: {(time.perf_counter()-t0)*1000:.0f}ms same={out2==out}')