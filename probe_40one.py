import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
M = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)
e = Engine(M, tier=OptLevel.AGGRESSIVE, device="cuda", layers=list(range(40)), tokenizer=tok)
ids = torch.randint(100, 1000, (1, 8), dtype=torch.long, device="cuda")
print("trying a single forward, chunk=8", flush=True)
lg = e.forward(ids, 0, chunk=8)
torch.cuda.synchronize()
print("OK logits=%s finite=%s" % (tuple(lg.shape), bool(torch.isfinite(lg).all())), flush=True)
