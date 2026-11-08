"""prefill+reset 之后，逐层逐组件找出哪一步开始坏。"""
import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
from tinyinfer.nn import rms_norm
M = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)
e = Engine(M, tier=OptLevel.AGGRESSIVE, device="cuda", layers=[0,1,2,3], tokenizer=tok)
L = torch.tensor([tok.encode("1+1 等于几？只回答数字。", add_special_tokens=False)], device="cuda")
S = torch.tensor([tok.encode("你好", add_special_tokens=False)], device="cuda")

e.reset()
e._forward_chunk(L, 0)
torch.cuda.synchronize()
e.reset()

h = e.embed[S.cpu()].to(e.device) if e.embed_on_cpu else e.embed[S]
h = h.unsqueeze(2).repeat(1, 1, e.cfg.hc_mult, 1)
pm = torch.zeros(1, 1, e.cfg.hc_mult, dtype=torch.float32, device=h.device)
pm[:, :, 0] = 1.0
hid = e.hashes(S.reshape(-1).tolist(), 0) if e.hashes is not None else None

def st(t, x):
    f = x.float()
    print("  %-16s finite=%-5s absmax=%s" % (t, bool(torch.isfinite(f).all()),
          ("%.3f" % float(f.abs().max())) if torch.isfinite(f).all() else "nan"), flush=True)

for i in range(4):
    b = e.block(i)
    if getattr(b, "engram", None) is not None and hid is not None:
        h = b.engram.forward(h, hid[b.engram.hash_index])
    apre, apost, acomb = b.hc_mixes(h.float(), b.hc_attn_fn, b.hc_attn_scale, b.hc_attn_base)
    hn = rms_norm(b.hc_pre(h, pm), b.attn_norm, e.cfg.rms_norm_eps)
    ya = b.attn.forward(hn, 0)
    st("L%d attn_out" % i, ya)
    x1 = b.hc_post(ya, h, apost, acomb)
    st("L%d hcpost1" % i, x1)
    fpre, fpost, fcomb = b.hc_mixes(x1.float(), b.hc_ffn_fn, b.hc_ffn_scale, b.hc_ffn_base)
    hn2 = rms_norm(b.hc_pre(x1, apre), b.ffn_norm, e.cfg.rms_norm_eps)
    st("L%d ffn_in" % i, hn2)
    yf = b.ffn.forward(hn2)
    st("L%d ffn_out" % i, yf)
    h = b.hc_post(yf, x1, fpost, fcomb)
    st("L%d out" % i, h)
    if not torch.isfinite(h).all():
        break
