import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
M = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)
e = Engine(M, tier=OptLevel.AGGRESSIVE, device="cuda", layers=[0,1,2,3], tokenizer=tok)

L = torch.tensor([tok.encode("1+1 等于几？只回答数字。", add_special_tokens=False)], device="cuda")
S = torch.tensor([tok.encode("你好", add_special_tokens=False)], device="cuda")
print("L=%d S=%d" % (L.shape[1], S.shape[1]), flush=True)

e.reset()
lg1 = e._forward_chunk(L, 0)
torch.cuda.synchronize()
print("step1 long finite=%s" % bool(torch.isfinite(lg1).all()), flush=True)

e.reset()
print("reset done", flush=True)

lg2 = e._forward_chunk(S, 0)
torch.cuda.synchronize()
print("step2 short finite=%s absmax=%s" % (
      bool(torch.isfinite(lg2).all()),
      ("%.3f" % float(lg2.float().abs().max())) if torch.isfinite(lg2).all() else "nan"), flush=True)
