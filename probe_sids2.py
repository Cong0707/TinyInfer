"""reset 后单行 decode：sids 里有几个无效槽位。"""
import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
import tinyinfer.deepseek.moe as MOE
M = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)
t = "1+1 等于几？只回答数字。请仔细思考后再回答。"
ids = torch.tensor([tok.encode(t, add_special_tokens=False)], device="cuda")
e = Engine(M, tier=OptLevel.AGGRESSIVE, device="cuda", layers=[0,1,2,3,4], tokenizer=tok)
moe = e.block(0).ffn

orig = moe._forward_gpu_only
def hook(x, xf, indices, weights):
    out = orig(x, xf, indices, weights)
    torch.cuda.synchronize()
    keep = getattr(moe, "_launch_keep", None)
    if keep is not None:
        sids_t = keep[0]
        s = sids_t.tolist()
        bad = [v for v in s if v < 0 or v > 100000]
        print("rows=%d sids=%s  bad=%d  out_finite=%s" % (
              int(xf.size(0)), s[:8], len(bad),
              bool(torch.isfinite(out).all())), flush=True)
    else:
        print("rows=%d no launch_keep out_finite=%s" % (
              int(xf.size(0)), bool(torch.isfinite(out).all())), flush=True)
    return out
moe._forward_gpu_only = hook

e.reset()
e._forward_chunk(ids, 0)
torch.cuda.synchronize()
e.reset()
lg = e._forward_chunk(ids[:, :1], 0)
torch.cuda.synchronize()
print("decode finite=%s" % bool(torch.isfinite(lg).all()), flush=True)
