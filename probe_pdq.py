"""prefill 之后 decode 出 NaN：加 reset 能否恢复。"""
import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
M = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)
t = "1+1 等于几？只回答数字。请仔细思考后再回答。"
ids = torch.tensor([tok.encode(t, add_special_tokens=False)], device="cuda")
N = ids.shape[1]
e = Engine(M, tier=OptLevel.AGGRESSIVE, device="cuda", layers=[0,1,2,3,4], tokenizer=tok)

mode = sys.argv[1]
e.reset()
la = e._forward_chunk(ids, 0)
torch.cuda.synchronize()
print("prefill finite=%s" % bool(torch.isfinite(la).all()), flush=True)

if mode == "reset":
    e.reset()
elif mode == "sync":
    torch.cuda.synchronize()
elif mode == "sync-reset":
    torch.cuda.synchronize(); e.reset()

lg = e._forward_chunk(ids[:, :1], 0)
torch.cuda.synchronize()
f = lg.float()
ok = bool(torch.isfinite(f).all())
print("MODE=%-11s decode finite=%-5s absmax=%s" % (
      mode, ok, ("%.3f" % float(f.abs().max())) if ok else "nan"), flush=True)
