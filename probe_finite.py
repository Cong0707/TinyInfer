"""n=11 输出 finite=False：是显存耗尽导致的分页，还是本来就错？"""
import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
M = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)

for n in (1, 2, 4, 8, 11):
    e = Engine(M, tier=OptLevel.AGGRESSIVE, device="cuda", layers=list(range(n)), tokenizer=tok)
    ids = torch.randint(100, 1000, (1, 4), dtype=torch.long, device="cuda")
    try:
        lg = e.forward(ids, 0, chunk=4)
        torch.cuda.synchronize()
        f = torch.isfinite(lg)
        am = float(lg.float().abs().max())
        print("n=%2d finite=%s absmax=%.3f" % (n, bool(f.all()), am), flush=True)
    except Exception as ex:
        print("n=%2d FAIL %s" % (n, str(ex)[:70]), flush=True)
    try:
        e._engine_lock.release()
    except Exception:
        pass
    del e
    import gc; gc.collect(); torch.cuda.empty_cache()
