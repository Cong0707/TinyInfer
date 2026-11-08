"""40 层全模型上 chunk 的真实上限（4 层的数据不代表 40 层）。"""
import sys, time, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
MODEL = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
e = Engine(MODEL, tier=OptLevel.AGGRESSIVE, device="cuda",
           layers=list(range(40)), tokenizer=tok)
print("planned chunk =", e.prefill_chunk, flush=True)
ids = torch.randint(100, 1000, (1, 64), dtype=torch.long, device="cuda")
for chunk in (4, 8, 16, 32):
    try:
        e.reset()
        torch.cuda.empty_cache()
        t0 = time.perf_counter()
        lg = e.forward(ids, 0, chunk=chunk)
        dt = time.perf_counter() - t0
        peak = torch.cuda.max_memory_allocated() / 2**20
        print("chunk=%2d OK 64tok %.1fs = %.1f tok/s peak=%.0fMB" %
              (chunk, dt, 64 / dt, peak), flush=True)
        torch.cuda.reset_peak_memory_stats()
    except Exception as ex:
        print("chunk=%2d FAIL %s" % (chunk, str(ex)[:80]), flush=True)
        torch.cuda.empty_cache()
