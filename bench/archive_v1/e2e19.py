import sys, time, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel

torch.manual_seed(0)
tok_dir = r"E:\Develop\Deepseek\tinyinfer\cache\weights"
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(tok_dir, trust_remote_code=True)
print("tokenizer ok, vocab:", len(tok))

eng = Engine(tok_dir, tier=OptLevel.BALANCED, device="cuda", layers=list(range(19)))
ids = tok("Hello, I am DeepSeek.", return_tensors="pt").input_ids[0].tolist()
print("prompt ids:", ids[:8], "...")
t0 = time.time()
outs = list(eng.stream_generate(ids, max_new_tokens=16, temperature=0.0))
dt = time.time() - t0
print("generated ids:", outs)
print("text:", repr(tok.decode(outs)))
print(f"time: {dt:.1f}s ({len(outs)/dt:.2f} tok/s)")
