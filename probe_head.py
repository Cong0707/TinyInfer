"""head 阶段：position 10 的 hidden 和 logits。"""
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
lg = e._forward_chunk(ids, 0)
torch.cuda.synchronize()
lh = e._last_hidden
print("last_hidden shape=%s" % (tuple(lh.shape),))
f = lh.float()
# 逐个 position 看 4 个 hc 分量
for p in range(lh.shape[1]):
    v = f[0, p]
    print("  pos %2d: finite=%-5s absmax=%10.3f nan=%d" % (
          p, bool(torch.isfinite(v).all()), float(v.abs().max()), int(torch.isnan(v).sum())), flush=True)
