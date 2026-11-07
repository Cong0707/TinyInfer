
import torch, collections
import sys
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
d = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
ids = tok("Hello", return_tensors="pt").input_ids[0].tolist()
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device="cuda", layers=list(range(40)))
from torch.profiler import profile, ProfilerActivity
with torch.inference_mode():
    list(eng.stream_generate(ids, max_new_tokens=3, temperature=0.0))
    gen = eng.stream_generate(ids, max_new_tokens=3, temperature=0.0)
    next(gen)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        next(gen)
        torch.cuda.synchronize()
kern = [e for e in prof.key_averages() if e.self_device_time_total > 0]
total = sum(e.count for e in kern)
print(f"一个 decode step 的 GPU 操作数: {total}")
print(f"  其中 kernel: {sum(e.count for e in kern if 'Memcpy' not in e.key and 'Memset' not in e.key)}")
print(f"  拷贝/清屏  : {sum(e.count for e in kern if 'Memcpy' in e.key or 'Memset' in e.key)}")
print(f"不同 kernel 种类: {len(kern)}")
print("--- 按次数 ---")
for e in sorted(kern, key=lambda x: -x.count)[:10]:
    print(f"  {e.count:5d} x  {e.key[:62]}")
