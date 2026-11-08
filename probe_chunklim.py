import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
MODEL = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
e = Engine(MODEL, tier=OptLevel.AGGRESSIVE, device="cuda",
           layers=list(range(1)), tokenizer=tok)
ids = torch.randint(100, 1000, (1, 16), dtype=torch.long, device="cuda")
for chunk in (1, 2, 4, 8, 16):
    try:
        e.reset()
        lg = e.forward(ids, 0, chunk=chunk)
        print("chunk=%2d OK logits=%s" % (chunk, tuple(lg.shape)), flush=True)
    except Exception as ex:
        print("chunk=%2d FAIL %s" % (chunk, str(ex)[:90]), flush=True)
        break
