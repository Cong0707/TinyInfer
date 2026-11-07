import sys, torch, time
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
d = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
p = tok("Hello", return_tensors="pt").input_ids[0].tolist()
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device="cuda", layers=list(range(40)))
with torch.inference_mode():
    eng.reset(); t0=time.time()
    list(eng.stream_generate(p, max_new_tokens=1, temperature=0.0))
    print(f"first: {time.time()-t0:.1f}s", flush=True)
    eng.reset(); t1=time.time()
    out=list(eng.stream_generate(p, max_new_tokens=4, temperature=0.0))
    dt=(time.time()-t1)/4
    print(f"FINAL {dt*1000:.0f} ms/tok = {1/dt:.2f} tok/s out={out}", flush=True)
