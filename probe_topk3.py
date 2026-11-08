"""topk 4 vs 6：40 层模型的单步耗时与输出差异。"""
import sys, time, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
M = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)
topk = int(sys.argv[1])

import os
os.environ["TI_TOP_K"] = str(topk)
e = Engine(M, tier=OptLevel.AGGRESSIVE, device="cuda", layers=list(range(40)), tokenizer=tok)
if topk != 6:
    for i in range(40):
        try:
            e.block(i).ffn.gate.topk = topk
        except Exception:
            pass
base = tok.encode("你好", add_special_tokens=False)
ids = torch.full((1, 1), base[0], dtype=torch.long, device="cuda")
pos = torch.ones(1, dtype=torch.long, device="cuda")
e.reset()
e._forward_chunk(ids, 1, positions=pos)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(3):
    lg = e._forward_chunk(ids, 1, positions=pos)
torch.cuda.synchronize()
ms = (time.perf_counter() - t0) / 3 * 1e3
f = lg.float()
print("RESULT topk=%d  %.0f ms/step  %.2f tok/s  argmax=%d  finite=%s" % (
      topk, ms, 1/(ms/1e3), int(f[0,-1].argmax()), bool(torch.isfinite(f).all())), flush=True)
