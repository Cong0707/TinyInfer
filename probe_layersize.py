"""一层常驻到底占多少显存：把 LAYER_SLACK 校准到实测。"""
import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
GB = 2**30

def alloc_gb():
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / GB

from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
M = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)

print("baseline %.3fGB" % alloc_gb(), flush=True)

# 逐层增加，看每加一层涨多少
prev = alloc_gb()
for n in (1, 2, 3, 4, 6):
    e = Engine(M, tier=OptLevel.AGGRESSIVE, device="cuda",
               layers=list(range(n)), tokenizer=tok)
    now = alloc_gb()
    print("layers=%d  alloc=%.3fGB  (+%.3fGB, %.1fMB/layer)  plan_layers=%d" % (
          n, now, now - prev, (now - prev) * 1024 / max(1, n - (0 if prev < 0.1 else 0)),
          e.hw_plan.gpu_layers), flush=True)
    prev = now
    try:
        e._engine_lock.release()
    except Exception:
        pass
    del e
    import gc; gc.collect(); torch.cuda.empty_cache()
    prev = alloc_gb()
