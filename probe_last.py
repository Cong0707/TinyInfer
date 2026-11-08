"""last_hidden 是否 finite：NaN 是在 head 之前还是之后。"""
import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
M = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)
ids = torch.tensor([tok.encode("1+1 等于几？只回答数字。", add_special_tokens=False)], device="cuda")
e = Engine(M, tier=OptLevel.AGGRESSIVE, device="cuda", layers=[0,1,2,3,4], tokenizer=tok)
e.reset()
e._dump_h = []
lg = e._forward_chunk(ids, 0)
torch.cuda.synchronize()
lh = getattr(e, "_last_hidden", None)
print("last_hidden:", None if lh is None else (tuple(lh.shape), bool(torch.isfinite(lh).all()),
      float(lh.float().abs().max()) if torch.isfinite(lh).all() else "nan"))
f = lg.float()
print("logits finite=%s nan=%d" % (bool(torch.isfinite(f).all()), int(torch.isnan(f).sum())))
# 逐位置看 logits
if not torch.isfinite(f).all():
    bad = torch.isnan(f).any(dim=-1)
    print("positions with NaN:", bad.nonzero().flatten().tolist())
for i, hin, hout in e._dump_h:
    print("  L%d out=%s" % (i, "finite" if torch.isfinite(hout).all() else "NAN"))
