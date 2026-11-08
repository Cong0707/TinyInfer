"""并发是否真的提升吞吐：同一引擎，batch=1 vs batch=B 的 forward 耗时。"""
import sys, time, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
M = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)
e = Engine(M, tier=OptLevel.AGGRESSIVE, device="cuda", layers=[0,1,2,3], tokenizer=tok)
base = tok.encode("你好", add_special_tokens=False)

def bench(rows, iters=5):
    ids = torch.full((rows, 1), base[0], dtype=torch.long, device="cuda")
    pos = torch.ones(rows, dtype=torch.long, device="cuda")
    # warmup
    e.reset()
    for _ in range(2):
        e._forward_chunk(ids, 1, positions=pos)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        e._forward_chunk(ids, 1, positions=pos)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    return dt * 1e3

print("rows  ms/step  tok/s  speedup")
base_ms = None
for r in (1, 2, 4, 8):
    try:
        ms = bench(r)
        if base_ms is None:
            base_ms = ms
        thr = r / (ms / 1e3)
        print("  %d   %7.1f   %8.1f   %5.2fx" % (r, ms, thr, (base_ms / ms) * r if base_ms else 1), flush=True)
    except Exception as ex:
        print("  %d   FAIL %s" % (r, str(ex)[:60]), flush=True)
