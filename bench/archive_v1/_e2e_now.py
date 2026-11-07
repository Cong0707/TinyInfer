
import sys, time, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer

d = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
print("tokenizer ok, vocab:", len(tok))
ids = tok("Hello, I am DeepSeek.", return_tensors="pt").input_ids[0].tolist()
print("prompt ids:", ids[:10])
eng = Engine(d, tier=OptLevel.BALANCED, device="cuda", layers=list(range(19)))
with torch.inference_mode():
    t0 = time.time()
    outs = list(eng.stream_generate(ids, max_new_tokens=16, temperature=0.0))
    dt = time.time() - t0
print("generated:", outs[:20])
print("text:", repr(tok.decode(outs)))
print(f"time {dt:.1f}s ({len(outs)/dt:.2f} tok/s)")
