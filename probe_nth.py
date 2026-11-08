"""第几次 prefill 之后短输入开始 NaN。"""
import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
M = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)
qs = ["1+1 等于几？只回答数字。",
      "2+2 等于几？只回答数字。",
      "3+3 等于几？只回答数字。",
      "4+4 等于几？只回答数字。"]
short = torch.tensor([tok.encode("你好", add_special_tokens=False)], device="cuda")
e = Engine(M, tier=OptLevel.AGGRESSIVE, device="cuda", layers=[0,1,2,3], tokenizer=tok)

for i, q in enumerate(qs):
    e.reset()
    ids = torch.tensor([tok.encode(q, add_special_tokens=False)], device="cuda")
    lg = e._forward_chunk(ids, 0)
    torch.cuda.synchronize()
    ok = bool(torch.isfinite(lg).all())
    # 紧接着短输入
    e.reset()
    lg2 = e._forward_chunk(short, 0)
    torch.cuda.synchronize()
    ok2 = bool(torch.isfinite(lg2).all())
    am2 = float(lg2.float().abs().max()) if ok2 else float("nan")
    print("after prefill#%d: long_finite=%-5s short_finite=%-5s short_absmax=%s" % (
          i + 1, ok, ok2, ("%.2f" % am2) if ok2 else "nan"), flush=True)
