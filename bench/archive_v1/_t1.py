import sys, torch, time
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
d = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
p = tok("Hello", return_tensors="pt").input_ids[0].tolist()
print("load", flush=True)
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device="cuda", layers=[0])
print("loaded", flush=True)
with torch.inference_mode():
    eng.reset()
    t0=time.time()
    out=list(eng.stream_generate(p, max_new_tokens=3, temperature=0.0))
    print(f"1layer: {(time.time()-t0)/3*1000:.0f} ms/tok out={out}", flush=True)
