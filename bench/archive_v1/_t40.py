import sys, torch, time, faulthandler
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
faulthandler.dump_traceback_later(60, repeat=False)
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
d = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
p = tok("Hello", return_tensors="pt").input_ids[0].tolist()
print("load", flush=True)
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device="cuda", layers=list(range(40)))
print("loaded", flush=True)
with torch.inference_mode():
    eng.reset()
    print("forward...", flush=True)
    out=list(eng.stream_generate(p, max_new_tokens=1, temperature=0.0))
    print(f"DONE {out}", flush=True)
