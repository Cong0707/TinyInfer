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
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    list(eng.stream_generate(p, max_new_tokens=1, temperature=0.0))
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    print(f'RESULT step={1000*(t1-t0):.0f}ms peak_alloc={torch.cuda.max_memory_allocated()/2**30:.2f}GB')
    # 打印 plan
    hp = eng.hw_plan
    for f in ('gpu_layers','stream_layers','slots','cache_bytes','buf'):
        v = getattr(hp, f, None)
        if v is not None: print(f'  {f} = {v}')