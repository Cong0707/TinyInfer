"""reset 之后 MoE 是否正常。"""
import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
M = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)
t = "1+1 等于几？只回答数字。请仔细思考后再回答。"
ids = torch.tensor([tok.encode(t, add_special_tokens=False)], device="cuda")
e = Engine(M, tier=OptLevel.AGGRESSIVE, device="cuda", layers=[0,1,2,3,4], tokenizer=tok)

moe = e.block(2).ffn
orig = moe.forward
rec = []
def h(x, *a, **k):
    out = orig(x, *a, **k)
    torch.cuda.synchronize()
    rec.append((tuple(x.shape), bool(torch.isfinite(x).all()),
                bool(torch.isfinite(out).all()),
                float(out.float().abs().max()) if torch.isfinite(out).all() else None))
    return out
moe.forward = h

e.reset()
e._forward_chunk(ids, 0)
torch.cuda.synchronize()
rec.clear()
e.reset()
lg = e._forward_chunk(ids[:, :1], 0)
torch.cuda.synchronize()
print("decode finite=%s" % bool(torch.isfinite(lg).all()), flush=True)
for i, (sh, xin, xout, am) in enumerate(rec):
    print("  moe%d in%s finite_in=%s finite_out=%s absmax=%s" % (i, sh, xin, xout, am), flush=True)
