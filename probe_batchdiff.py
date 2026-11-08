"""拼批是否改变单行的数值：逐组件定位。"""
import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer

MODEL = r"E:\Models\DeepSeek-V4.1-Flash-1017".replace("Models", "models")
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

for nl in (1, 2):
    e = Engine(MODEL, tier=OptLevel.AGGRESSIVE, device="cuda",
               layers=list(range(nl)), tokenizer=tok)
    pad = getattr(e, "pad_id", 0) or 0
    ids = torch.full((2, 2), pad, dtype=torch.long, device="cuda")
    ids[0, 0] = 100
    ids[1, 0] = 200
    pos = torch.tensor([1, 2], dtype=torch.long, device="cuda")
    e.reset()
    one = e.forward(ids[:1], 0, positions=pos[:1]).float()
    e.reset()
    two = e.forward(ids, 0, positions=pos).float()
    d = float((two[0] - one[0]).abs().max())
    ok = bool(torch.isfinite(two).all())
    print("n_layers=%d  row0 maxdiff=%.4f  finite=%s" % (nl, d, ok), flush=True)
    try:
        e._engine_lock.release()
    except Exception:
        pass
    del e
    import gc; gc.collect(); torch.cuda.empty_cache()
