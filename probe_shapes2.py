"""分批时 indices/weights 的形状是否正确。"""
import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
from tinyinfer.nn import rms_norm
M = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)
base = "1+1 等于几？只回答数字。请仔细思考后再回答，不要着急。再加一些字凑够到三十个词左右吧。 "
enc = tok.encode(base, add_special_tokens=False)
ids = torch.tensor([enc[:11]], device="cuda")
e = Engine(M, tier=OptLevel.AGGRESSIVE, device="cuda", layers=[0,1,2], tokenizer=tok)

moe = e.block(2).ffn
orig = moe.gate.forward
def hooked(x, *a, **k):
    w, idx = orig(x, *a, **k)
    print("gate: x=%s  w=%s  idx=%s" % (tuple(x.shape), tuple(w.shape), tuple(idx.shape)), flush=True)
    return w, idx
moe.gate.forward = hooked

orig_fwd = moe._forward_gpu_only
def hook2(x, xf, indices, weights):
    print("  _forward_gpu_only: x=%s xf=%s idx=%s w=%s rows=%d nexp=%d" % (
          tuple(x.shape), tuple(xf.shape), tuple(indices.shape), tuple(weights.shape),
          xf.size(0), indices.numel()), flush=True)
    return orig_fwd(x, xf, indices, weights)
moe._forward_gpu_only = hook2

e.reset()
e._forward_chunk(ids, 0)
