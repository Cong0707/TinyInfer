"""第 1 层内部：哪个 hc_post 把输出放大。"""
import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
M = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)
t = "1+1 等于几？只回答数字。"
ids = torch.tensor([tok.encode(t, add_special_tokens=False)], device="cuda")
e = Engine(M, tier=OptLevel.AGGRESSIVE, device="cuda", layers=[0,1], tokenizer=tok)
e.reset()
h = e.embed[ids.cpu()].to(e.device) if e.embed_on_cpu else e.embed[ids]
h = h.unsqueeze(2).repeat(1, 1, e.cfg.hc_mult, 1)
pm = torch.zeros(h.size(0), h.size(1), e.cfg.hc_mult, dtype=torch.float32, device=h.device)
pm[:, :, 0] = 1.0

def st(tag, x):
    f = x.float()
    print("  %-26s finite=%-5s absmax=%9.3f" % (tag, bool(torch.isfinite(f).all()), float(f.abs().max())))

b0 = e.block(0)
h, pm = b0.forward(h, 0, pm)
st("L1 input (after L0)", h)

b = e.block(1)
st("L1 x", h)
apre, apost, acomb = b.hc_mixes(h, b.hc_attn_fn, b.hc_attn_scale, b.hc_attn_base)
st("attn_pre", apre); st("attn_post", apost); st("attn_comb", acomb)

from tinyinfer.nn import rms_norm
hh = rms_norm(b.hc_pre(h, pm), b.attn_norm, e.cfg.rms_norm_eps)
st("attn_norm out", hh)
ya = b.attn.forward(hh, 0)
st("y_attn", ya)
x1 = b.hc_post(ya, h, apost, acomb)
st("after hc_post #1", x1)

fpre, fpost, fcomb = b.hc_mixes(x1, b.hc_ffn_fn, b.hc_ffn_scale, b.hc_ffn_base)
hh2 = rms_norm(b.hc_pre(x1, apre), b.ffn_norm, e.cfg.rms_norm_eps)
st("ffn_norm out", hh2)
yf = b.ffn.forward(hh2)
st("y_ffn", yf)
x2 = b.hc_post(yf, x1, fpost, fcomb)
st("after hc_post #2", x2)
