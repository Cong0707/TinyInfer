import sys, time, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
MODEL = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
e = Engine(MODEL, tier=OptLevel.AGGRESSIVE, device="cuda",
           layers=list(range(4)), tokenizer=tok)
ids = torch.randint(100, 1000, (1, 128), dtype=torch.long, device="cuda")
for chunk in (8, 16, 32, 64, 128):
    e.reset()
    t0 = time.perf_counter()
    lg = e.forward(ids, 0, chunk=chunk)
    dt = time.perf_counter() - t0
    print("chunk=%3d  128 tok %.1fs = %.1f tok/s" % (chunk, dt, 128 / dt), flush=True)
