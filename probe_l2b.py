"""第 2 层内部：哪一步把 absmax 从 1 推到 940。"""
import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
from tinyinfer.nn import rms_norm
M = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)
t = "1+1 等于几？只回答数字。"
ids = torch.tensor([tok.encode(t, add_special_tokens=False)], device="cuda")
e = Engine(M, tier=OptLevel.AGGRESSIVE, device="cuda", layers=[0,1,2], tokenizer=tok)
e.reset()

h = e.embed[ids.cpu()].to(e.device) if e.embed_on_cpu else e.embed[ids]
h = h.unsqueeze(2).repeat(1, 1, e.cfg.hc_mult, 1)
pm = torch.zeros(h.size(0), h.size(1), e.cfg.hc_mult, dtype=torch.float32, device=h.device)
pm[:, :, 0] = 1.0

def st(tag, x):
    f = x.float()
    print("  %-22s finite=%-5s absmax=%9.3f absmin_nonzero=%s" % (
          tag, bool(torch.isfinite(f).all()), float(f.abs().max()),
          float(f[f != 0].abs().min()) if (f != 0).any() else 0.0))

for lid in (0, 1):
    b = e.block(lid)
    h, pm = b.forward(h, 0, pm)

b = e.block(2)
st("L2 input", h)
residual = h
hp = b.hc_pre(h, pm)
st("after hc_pre", hp)
hn = rms_norm(hp, b.attn_norm, e.cfg.rms_norm_eps)
st("after attn_norm", hn)
o = b.attn.forward(hn, 0)
st("after attn", o)

# 压缩件的中间量
a = b.attn
rt = getattr(a, "rt", None)
for k in ("compress_kv", "index_k", "topk_idxs", "candidates", "latent"):
    v = getattr(rt, k, None) if rt is not None else None
    if torch.is_tensor(v):
        st("rt." + k, v)
