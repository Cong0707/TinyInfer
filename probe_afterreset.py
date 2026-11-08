import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
M = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)
e = Engine(M, tier=OptLevel.AGGRESSIVE, device="cuda", layers=[0,1,2,3], tokenizer=tok)
base = tok.encode("1+1 等于几？只回答数字。", add_special_tokens=False)

for N in (1, 2, 3, 4, 6, 11):
    ids = torch.tensor([base[:N]], device="cuda")
    e.reset()
    lg = e._forward_chunk(ids, 0)
    torch.cuda.synchronize()
    ok = bool(torch.isfinite(lg).all())
    print("after reset: N=%2d finite=%-5s absmax=%s" % (
          N, ok, ("%.2f" % float(lg.float().abs().max())) if ok else "nan"), flush=True)
