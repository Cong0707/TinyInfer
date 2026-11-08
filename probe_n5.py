"""逐层正常但整段 NaN：问题在层循环之后。"""
import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
M = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)
t = "1+1 等于几？只回答数字。"
ids = torch.tensor([tok.encode(t, add_special_tokens=False)], device="cuda")
e = Engine(M, tier=OptLevel.AGGRESSIVE, device="cuda", layers=list(range(5)), tokenizer=tok)
e.reset()

# 用 hook 记录每层输出与最终 hidden
e._last_hidden = None
lg = e.forward(ids, 0)
torch.cuda.synchronize()
print("logits finite=%s" % bool(torch.isfinite(lg).all()), flush=True)
lh = getattr(e, "_last_hidden", None)
if lh is not None:
    print("last_hidden finite=%s absmax=%.3f shape=%s" % (
          bool(torch.isfinite(lh).all()), float(lh.float().abs().max()), tuple(lh.shape)), flush=True)
print("head dtype=%s device=%s" % (e.head.dtype, e.head.device), flush=True)
