
import sys, time, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
d = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
ids = tok("Hello", return_tensors="pt").input_ids[0].tolist()
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device="cuda", layers=list(range(40)))
with torch.inference_mode():
    list(eng.stream_generate(ids, max_new_tokens=4, temperature=0.0))
    marks=[]
    for _ in eng.stream_generate(ids, max_new_tokens=8, temperature=0.0):
        marks.append(time.time())
gaps = sorted(marks[i]-marks[i-1] for i in range(2,len(marks)))
print(f"STEP: {gaps[len(gaps)//2]*1000:.0f} ms/token")

# GPU 侧：单个 decode step 的 kernel 总时间
from torch.profiler import profile, ProfilerActivity
gen = eng.stream_generate(ids, max_new_tokens=3, temperature=0.0)
next(gen)
torch.cuda.synchronize()
try:
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        next(gen)
    torch.cuda.synchronize()
    tot = sum(e.self_device_time_total for e in prof.key_averages())
    print(f"GPU kernels in one step: {tot/1000:.1f} ms")
    top = sorted(((e.self_device_time_total/1000, e.key) for e in prof.key_averages()
                  if e.self_device_time_total>0), reverse=True)[:6]
    for ms,k in top: print(f"  {ms:8.2f} ms  {k[:64]}")
except Exception as e:
    print("profiler:", type(e).__name__, str(e)[:100])
