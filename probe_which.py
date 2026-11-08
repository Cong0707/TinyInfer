"""哪一层引入 NaN：按层数递增（每进程一个引擎）。"""
import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
M = r"E:\models\DeepSeek-V4.1-Flash-1017"

spec = sys.argv[1]
layers = [int(x) for x in spec.split(",")]
tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)
t = "1+1 等于几？只回答数字。"
ids = torch.tensor([tok.encode(t, add_special_tokens=False)], device="cuda")
e = Engine(M, tier=OptLevel.AGGRESSIVE, device="cuda", layers=layers, tokenizer=tok)
e.reset()
lg = e.forward(ids, 0)
torch.cuda.synchronize()
f = lg.float()
ok = bool(torch.isfinite(f).all())
lh = getattr(e, "_last_hidden", None)
lok = bool(torch.isfinite(lh).all()) if lh is not None else None
print("RESULT layers=%-12s logits_finite=%-5s hidden_finite=%s" % (
      spec, ok, lok), flush=True)
