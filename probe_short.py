"""短序列是否必然 NaN。"""
import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
M = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)
e = Engine(M, tier=OptLevel.AGGRESSIVE, device="cuda", layers=[0,1,2,3], tokenizer=tok)
base = tok.encode("1+1 等于几？只回答数字。请仔细思考。", add_special_tokens=False)
for N in (1, 2, 3, 4, 5, 6, 8):
    e.reset()
    ids = torch.tensor([base[:N]], device="cuda")
    lg = e._forward_chunk(ids, 0)
    torch.cuda.synchronize()
    f = lg.float()
    ok = bool(torch.isfinite(f).all())
    print("N=%d finite=%-5s absmax=%s" % (
          N, ok, ("%.2f" % float(f.abs().max())) if ok else "nan"), flush=True)
