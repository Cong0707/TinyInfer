"""批次之间 _sids_buf / _aq_buf / _ascb / _gpu_y_buf 的复用状态。"""
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
e = Engine(M, tier=OptLevel.AGGRESSIVE, device="cuda", layers=[0,1,2], tokenizer=tok)
moe = e.block(2).ffn

orig = moe._forward_gpu_only
def hook(x, xf, indices, weights):
    bufs = {}
    for name in ("_sids_buf", "_aq_buf", "_ascb", "_gpu_y_buf"):
        b = getattr(moe, name, None)
        bufs[name] = None if b is None else (tuple(b.shape), int(b.numel()))
    print("BATCH rows=%d nexp=%d  %s" % (xf.size(0), indices.numel(), bufs), flush=True)
    return orig(x, xf, indices, weights)
moe._forward_gpu_only = hook

e.reset()
lg = e._forward_chunk(ids, 0)
f = lg.float()
bad = torch.isnan(f).any(dim=-1)[0].nonzero().flatten().tolist()
print("bad=%s" % bad, flush=True)
