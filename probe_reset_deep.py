"""reset 后 decode：attention 到 MoE 之间哪里 NaN。"""
import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
from tinyinfer.nn import rms_norm
M = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)
t = "1+1 等于几？只回答数字。请仔细思考后再回答。"
ids = torch.tensor([tok.encode(t, add_special_tokens=False)], device="cuda")
e = Engine(M, tier=OptLevel.AGGRESSIVE, device="cuda", layers=[0,1,2,3,4], tokenizer=tok)

e.reset()
e._forward_chunk(ids, 0)
torch.cuda.synchronize()
e.reset()

# 逐层手调，看哪一层哪个环节 NaN
h = e.embed[ids[:, :1].cpu()].to(e.device) if e.embed_on_cpu else e.embed[ids[:, :1]]
h = h.unsqueeze(2).repeat(1, 1, e.cfg.hc_mult, 1)
pm = torch.zeros(1, 1, e.cfg.hc_mult, dtype=torch.float32, device=h.device)
pm[:, :, 0] = 1.0
hid = e.hashes(ids[:, :1].reshape(-1).tolist(), 0) if e.hashes is not None else None

def st(tag, x):
    f = x.float()
    print("  %-20s finite=%-5s absmax=%s" % (tag, bool(torch.isfinite(f).all()),
          ("%.4f" % float(f.abs().max())) if torch.isfinite(f).all() else "nan"), flush=True)

for i in range(5):
    b = e.block(i)
    if getattr(b, "engram", None) is not None and hid is not None:
        h = b.engram.forward(h, hid[b.engram.hash_index])
        st("L%d engram" % i, h)
    apre, apost, acomb = b.hc_mixes(h.float(), b.hc_attn_fn, b.hc_attn_scale, b.hc_attn_base)
    hn = rms_norm(b.hc_pre(h, pm), b.attn_norm, e.cfg.rms_norm_eps)
    st("L%d attn_in" % i, hn)
    ya = b.attn.forward(hn, 0)
    st("L%d attn_out" % i, ya)
    x1 = b.hc_post(ya, h, apost, acomb)
    st("L%d hcpost1" % i, x1)
    if not torch.isfinite(x1).all():
        print("  ^^ 这一层坏了", flush=True)
        break
    fpre, fpost, fcomb = b.hc_mixes(x1.float(), b.hc_ffn_fn, b.hc_ffn_scale, b.hc_ffn_base)
    hn2 = rms_norm(b.hc_pre(x1, apre), b.ffn_norm, e.cfg.rms_norm_eps)
    st("L%d ffn_in" % i, hn2)
    yf = b.ffn.forward(hn2)
    st("L%d ffn_out" % i, yf)
    h = b.hc_post(yf, x1, fpost, fcomb)
    st("L%d out" % i, h)
    if not torch.isfinite(h).all():
        print("  ^^ 这一层输出坏了", flush=True)
        break
