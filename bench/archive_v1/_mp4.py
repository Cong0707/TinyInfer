import sys, torch, os, time
sys.path.insert(0, r'E:\\Develop\\Deepseek\\tinyinfer\\src')
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
os.environ['TI_MTP'] = '1'
d = r'E:\\models\\DeepSeek-V4.1-Flash-1017'
tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
p = tok('The capital of France is', return_tensors='pt').input_ids[0].tolist()
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device='cuda', layers=list(range(10)))
print('RESULT mtp:', len(eng.mtp) if getattr(eng,'mtp',None) else 0)
print('RESULT vram:', round(torch.cuda.memory_allocated()/2**30, 2), 'GB')
with torch.inference_mode():
    eng.reset()
    out = eng.speculative_generate(p, max_new_tokens=10, temperature=0.0)
    toks, rate = out if isinstance(out, tuple) else (out, None)
    print(f'RESULT accept_rate={rate}')
    print(f'RESULT tokens={toks}')
    eng.reset()
    t0 = time.perf_counter()
    base = list(eng.stream_generate(p, max_new_tokens=10, temperature=0.0))
    torch.cuda.synchronize()
    tb = time.perf_counter()-t0
    print(f'RESULT base={base}')
    print(f'RESULT match {sum(1 for i in range(min(len(toks),len(base))) if toks[i]==base[i])}/{min(len(toks),len(base))}  base_time={tb*1000:.0f}ms')