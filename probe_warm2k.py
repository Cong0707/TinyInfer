"""新引擎直接 2048 vs 先热身再 2048。"""
import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
M = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)
e = Engine(M, tier=OptLevel.AGGRESSIVE, device="cuda", layers=[0,1,2,3], tokenizer=tok)
base = tok.encode("你好今天天气不错", add_special_tokens=False)
long_ids = torch.tensor([(base*2000)[:2048]], device="cuda")

mode = sys.argv[1]
if mode == "warm":
    for w in (64, 256, 512, 1024):
        e.reset()
        e._forward_chunk(torch.tensor([(base*2000)[:w]], device="cuda"), 0)
    torch.cuda.synchronize()

e.reset()
lg = e._forward_chunk(long_ids, 0)
torch.cuda.synchronize()
f = lg.float()
ok = bool(torch.isfinite(f).all())
print("RESULT mode=%-5s finite=%s absmax=%s" % (mode, ok,
      ("%.2f" % float(f.abs().max())) if ok else "nan"), flush=True)
