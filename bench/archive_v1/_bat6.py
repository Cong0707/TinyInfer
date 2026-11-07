
import sys, torch, traceback, time
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from tinyinfer.sched import BatchScheduler, Request
from transformers import AutoTokenizer

d = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device="cuda", layers=list(range(40)))
print("pad_id:", eng.pad_id)
p1 = tok("Hello", return_tensors="pt").input_ids[0].tolist()
p2 = tok("The capital of France is", return_tensors="pt").input_ids[0].tolist()

with torch.inference_mode():
    eng.reset()
    a_only = list(eng.stream_generate(p1, max_new_tokens=6, temperature=0.0))
print("单序列 A:", a_only)

s = BatchScheduler(eng, max_batch=2)
s.submit(Request(prompt=p1, max_new_tokens=6, temperature=0.0))
s.submit(Request(prompt=p2, max_new_tokens=6, temperature=0.0))
t0 = time.perf_counter()
try:
    reqs = s.run()
    dt = time.perf_counter() - t0
    print(f"批2 {dt:.2f}s {s.stats()}")
    for r in reqs:
        print("  out:", r.out, " matchA:", r.out == a_only if r.prompt == p1 else "-")
except Exception:
    traceback.print_exc()
