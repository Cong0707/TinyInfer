
import sys, time, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer

d = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
ids = tok("Hello, I am DeepSeek.", return_tensors="pt").input_ids[0].tolist()
eng = Engine(d, tier=OptLevel.BALANCED, device="cuda", layers=list(range(40)))

with torch.inference_mode():
    list(eng.stream_generate(ids, max_new_tokens=2, temperature=0.0))   # warm

from torch.profiler import profile, ProfilerActivity
with torch.inference_mode():
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        list(eng.stream_generate(ids, max_new_tokens=3, temperature=0.0))

# 统计实际执行到的算子
from collections import Counter
c = Counter()
for e in prof.key_averages():
    name = e.key
    if name.startswith("_C::") or name.startswith("_moe_C::"):
        c[name] += e.count
    elif "::" in name:
        ns = name.split("::")[0]
        if ns in ("_C", "_moe_C", "vllm"):
            c[name] += e.count
print("=== 本次 decode 实际执行的 vLLM 算子 ===")
if not c:
    print("(profiler 没抓到，看下面的全部 top 算子)")
for k, v in c.most_common():
    print(f"  {k:<60} x{v}")
print()
print("=== top 20 算子（含 torch 原生）===")
for e in prof.key_averages()[:20]:
    if e.count:
        print(f"  {e.key[:58]:<58} x{e.count:<6} cuda {e.self_device_time_total/1000:>8.2f}ms")
