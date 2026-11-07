
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
    list(eng.stream_generate(ids, max_new_tokens=4, temperature=0.0))  # warm
    # 精确框住一个 decode step
    gen = eng.stream_generate(ids, max_new_tokens=6, temperature=0.0)
    next(gen)                      # prefill + 第 1 个 token
    t0 = time.time()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        next(gen)                  # 单个 decode step
        torch.cuda.synchronize()
    wall_ms = (time.time() - t0) * 1000
    # 再测几个 step 的均值
    for _ in range(2): next(gen)
    t1 = time.time()
    for _ in range(3): next(gen)
    avg_ms = (time.time() - t1) / 3 * 1000
gpu_ms = sum(e.self_device_time_total for e in prof.key_averages()) / 1000
print(f"decode step wall : {avg_ms:.0f} ms/token")
print(f"  profiled step  : {wall_ms:.0f} ms")
print(f"  GPU kernels    : {gpu_ms:.0f} ms")
print(f"  非 GPU（IO等）  : {wall_ms - gpu_ms:.0f} ms  ({100*(wall_ms-gpu_ms)/wall_ms:.0f}%)")
top = sorted(((e.self_device_time_total/1000, e.key) for e in prof.key_averages()
              if e.self_device_time_total > 0), reverse=True)[:8]
print("--- 最耗时的 GPU kernel ---")
for ms, k in top:
    print(f"  {ms:7.1f} ms  {k[:70]}")
