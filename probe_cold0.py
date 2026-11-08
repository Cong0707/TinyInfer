"""cold 时 L0 的输入与 attention 内部。"""
import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
from tinyinfer.nn import rms_norm
M = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)
e = Engine(M, tier=OptLevel.AGGRESSIVE, device="cuda", layers=[0], tokenizer=tok)
base = tok.encode("你好今天天气不错", add_special_tokens=False)
N = 2048
ids = torch.tensor([(base*2000)[:N]], device="cuda")
e.reset()
b = e.block(0)
h = e.embed[ids.cpu()].to(e.device) if e.embed_on_cpu else e.embed[ids]
h = h.unsqueeze(2).repeat(1, 1, e.cfg.hc_mult, 1)
pm = torch.zeros(1, N, e.cfg.hc_mult, dtype=torch.float32, device=h.device)
pm[:, :, 0] = 1.0

def st(t, x):
    f = x.float()
    ok = bool(torch.isfinite(f).all())
    print("  %-16s finite=%-5s absmax=%s" % (t, ok,
          ("%.3f" % float(f.abs().max())) if ok else "nan"), flush=True)
    return ok

st("embed", h)
hid = e.hashes(ids.reshape(-1).tolist(), 0) if e.hashes is not None else None
if getattr(b, "engram", None) is not None and hid is not None:
    h = b.engram.forward(h, hid[b.engram.hash_index])
    st("engram", h)
apre, apost, acomb = b.hc_mixes(h.float(), b.hc_attn_fn, b.hc_attn_scale, b.hc_attn_base)
st("hc_mixes", acomb)
pre = b.hc_pre(h, pm)
st("hc_pre", pre)
hn = rms_norm(pre, b.attn_norm, e.cfg.rms_norm_eps)
st("attn_in", hn)
pos = torch.arange(N, device="cuda").unsqueeze(0)
ya = b.attn.forward(hn, 0, positions=pos)
st("attn_out", ya)
