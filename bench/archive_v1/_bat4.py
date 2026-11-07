
import sys, torch, time
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

# 单序列基线
with torch.inference_mode():
    eng.reset()
    out1 = list(eng.stream_generate(p1, max_new_tokens=8, temperature=0.0))
print("单序列 A:", out1)
print("单序列 A text:", tok.decode(out1))

# 批 2
s = BatchScheduler(eng, max_batch=2)
s.submit(Request(prompt=p1, max_new_tokens=8, temperature=0.0))
s.submit(Request(prompt=p2, max_new_tokens=8, temperature=0.0))
t0 = time.perf_counter()
reqs = s.run()
dt = time.perf_counter() - t0
print(f"批2 用时 {dt:.2f}s  stats={s.stats()}")
print("批2 A:", reqs[0].out if reqs else "?", " match:", reqs[0].out == out1 if reqs else "?")
print("批2 B:", reqs[1].out if len(reqs)>1 else "?")
print("批2 B text:", tok.decode(reqs[1].out) if len(reqs)>1 else "?")
