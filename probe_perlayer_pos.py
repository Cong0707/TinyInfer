"""逐层看 last_hidden 的 pos 9/10 何时开始坏。"""
import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
M = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)
base = "1+1 等于几？只回答数字。请仔细思考后再回答，不要着急。再加一些字凑够到三十个词左右吧。 "
enc = tok.encode(base, add_special_tokens=False)
ids = torch.tensor([enc[:11]], device="cuda")
e = Engine(M, tier=OptLevel.AGGRESSIVE, device="cuda", layers=[0,1,2,3,4], tokenizer=tok)
e.reset()
e._dump_h = []
e._forward_chunk(ids, 0)
torch.cuda.synchronize()
for i, hin, hout in e._dump_h:
    f = hout.float()
    vals = []
    for p in (8, 9, 10):
        v = f[0, p]
        vals.append("p%d=%s" % (p, ("nan" if not torch.isfinite(v).all() else "%.2f" % float(v.abs().max()))))
    print("L%d: %s" % (i, "  ".join(vals)), flush=True)
