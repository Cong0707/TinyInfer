import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer

MODEL = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
e = Engine(MODEL, tier=OptLevel.AGGRESSIVE, device="cuda", layers=[0], tokenizer=tok)
e.reset()
pad = getattr(e, "pad_id", 0) or 0
w = 2
ids = torch.full((2, w), pad, dtype=torch.long, device="cuda")
ids[0, 0] = 100
ids[1, 0] = 200
pos = torch.tensor([0, 4], dtype=torch.long, device="cuda")

# 单行对照
e.reset()
one = e.forward(ids[:1], 0, positions=pos[:1])
print("single finite:", bool(torch.isfinite(one).all()), "absmax=%.3f" % float(one.abs().max()), flush=True)

e.reset()
two = e.forward(ids, 0, positions=pos)
print("batch2 finite:", bool(torch.isfinite(two).all()), "absmax=%.3f" % float(two.abs().max()), flush=True)
print("row0 finite:", bool(torch.isfinite(two[0]).all()), flush=True)
print("row1 finite:", bool(torch.isfinite(two[1]).all()), flush=True)
