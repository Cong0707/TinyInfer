"""不拦截 vs 拦截：确认 L4 NaN 是否与 monkey-patch 有关。"""
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
lg = e._forward_chunk(ids, 0)
torch.cuda.synchronize()
f = lg.float()
print("PLAIN n=5 finite=%s absmax=%.3f nan=%d" % (
      bool(torch.isfinite(f).all()),
      float(f.abs().max()) if torch.isfinite(f).all() else float("nan"),
      int(torch.isnan(f).sum())), flush=True)
