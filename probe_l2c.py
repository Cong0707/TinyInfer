"""第 2 层：attn(8.9) 到块输出(940) 之间发生了什么。"""
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

def st(tag, x):
    f = x.float()
    print("  %-22s finite=%-5s absmax=%10.3f" % (tag, bool(torch.isfinite(f).all()), float(f.abs().max())))

# 用 _forward_chunk 自己的 h，保证与引擎一致
h = e.embed[ids.cpu()].to(e.device) if e.embed_on_cpu else e.embed[ids]
h = h.unsqueeze(2).repeat(1, 1, e.cfg.hc_mult, 1)
pm = torch.zeros(h.size(0), h.size(1), e.cfg.hc_mult, dtype=torch.float32, device=h.device)
pm[:, :, 0] = 1.0
hid = e.hashes(ids.reshape(-1).tolist(), 0) if e.hashes is not None else None
for i in (0, 1):
    b = e.block(i)
    if getattr(b, "engram", None) is not None and hid is not None:
        h = b.engram.forward(h, hid[b.engram.hash_index])
    h, pm = b.forward(h, 0, pm)

b = e.block(2)
st("L2 input", h)
residual = h
apre, apost, acomb = b.hc_mixes(h, b.hc_attn_fn, b.hc_attn_scale, b.hc_attn_base)
h_n = rms_norm(b.hc_pre(h, pm), b.attn_norm, e.cfg.rms_norm_eps)
st("attn_norm", h_n)
ya = b.attn.forward(h_n, 0)
st("y_attn", ya)
x1 = b.hc_post(ya, residual, apost, acomb)
st("hc_post#1", x1)

residual = x1
fpre, fpost, fcomb = b.hc_mixes(x1, b.hc_ffn_fn, b.hc_ffn_scale, b.hc_ffn_base)
h_n2 = rms_norm(b.hc_pre(x1, apre), b.ffn_norm, e.cfg.rms_norm_eps)
st("ffn_norm", h_n2)
yf = b.ffn.forward(h_n2)
st("y_ffn", yf)
x2 = b.hc_post(yf, residual, fpost, fcomb)
st("hc_post#2 (=block out)", x2)
