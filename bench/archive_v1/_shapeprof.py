
import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
from torch.profiler import profile, ProfilerActivity

d = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
ids = tok("Hello", return_tensors="pt").input_ids[0].tolist()
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device="cuda", layers=list(range(40)))
with torch.inference_mode():
    list(eng.stream_generate(ids, max_new_tokens=3, temperature=0.0))
    gen = eng.stream_generate(ids, max_new_tokens=3, temperature=0.0)
    next(gen); torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
        next(gen)
        torch.cuda.synchronize()

rows = []
for e in prof.key_averages(group_by_input_shape=True):
    ms = e.self_device_time_total / 1000
    if ms >= 0.3 and "Memcpy" not in e.key and "Memset" not in e.key:
        rows.append((ms, e.count, e.key[:58], str(e.input_shapes)[:60]))
rows.sort(reverse=True)
print(f"{'ms':>8} {'n':>5}  kernel / shapes")
for ms, cnt, k, sh in rows[:18]:
    print(f"{ms:8.2f} {cnt:5d}  {k}\n{'':14}{sh}")
