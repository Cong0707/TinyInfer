"""40 层模型（真实配置）的 decode 单步与并发。"""
import sys, time, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
M = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)
e = Engine(M, tier=OptLevel.AGGRESSIVE, device="cuda", layers=list(range(40)), tokenizer=tok)
print("plan:", e.hw_plan.reason, flush=True)
base = tok.encode("你好", add_special_tokens=False)

def bench(rows, iters=3):
    ids = torch.full((rows, 1), base[0], dtype=torch.long, device="cuda")
    pos = torch.ones(rows, dtype=torch.long, device="cuda")
    e.reset()
    for _ in range(1):
        e._forward_chunk(ids, 1, positions=pos)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        e._forward_chunk(ids, 1, positions=pos)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3

for rows in (1, 2):
    try:
        ms = bench(rows)
        print("rows=%d  %.1f ms/step  %.2f tok/s" % (rows, ms, rows / (ms/1e3)), flush=True)
        torch.cuda.empty_cache()
    except Exception as ex:
        print("rows=%d FAIL %s" % (rows, str(ex)[:70]), flush=True)
