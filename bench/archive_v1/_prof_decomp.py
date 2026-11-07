
import sys, time, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
from torch.profiler import profile, ProfilerActivity

d = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
ids = tok("Hello, I am DeepSeek.", return_tensors="pt").input_ids[0].tolist()
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device="cuda", layers=list(range(40)))
with torch.inference_mode():
    list(eng.stream_generate(ids, max_new_tokens=3, temperature=0.0))
    # 稳态计时
    marks=[]
    for _ in eng.stream_generate(ids, max_new_tokens=8, temperature=0.0):
        marks.append(time.time())
    gaps = sorted(marks[i]-marks[i-1] for i in range(2,len(marks)))
    step = gaps[len(gaps)//2]
    # GPU 侧总时间
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in eng.stream_generate(ids, max_new_tokens=3, temperature=0.0):
            pass
        torch.cuda.synchronize()
gpu_us = sum(e.self_device_time_total for e in prof.key_averages())
n_tok = 3
st = eng.expert_store.stats()
print(f"STEP ms/token      : {step*1000:.0f}")
print(f"GPU kernel ms/token: {gpu_us/n_tok/1000:.0f}")
print(f"  -> 非GPU(IO+同步)  : {step*1000 - gpu_us/n_tok/1000:.0f}")
print("hit_rate:", round(st["ram"]["hit_rate"],3), " disk_gb:", round(st["ram"]["disk_gb"],2))
print("reader GB/s:", round(st["reader"]["gbps"],2))
