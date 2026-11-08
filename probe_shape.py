"""x 和 xf 的真实形状：入口打点。"""
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

moe = e.block(2).ffn
orig = moe.forward
def hooked(x, *a, **k):
    print("moE.forward x.shape=%s" % (tuple(x.shape),), flush=True)
    print("  x.view(-1,dim) -> %s" % (tuple(x.view(-1, moe.cfg.dim).shape),), flush=True)
    return orig(x, *a, **k)
moe.forward = hooked

h = e.embed[ids.cpu()].to(e.device) if e.embed_on_cpu else e.embed[ids]
h = h.unsqueeze(2).repeat(1, 1, e.cfg.hc_mult, 1)
pm = torch.zeros(h.size(0), h.size(1), e.cfg.hc_mult, dtype=torch.float32, device=h.device)
pm[:, :, 0] = 1.0
hid = e.hashes(ids.reshape(-1).tolist(), 0) if e.hashes is not None else None
for i in (0, 1):
    bb = e.block(i)
    if getattr(bb, "engram", None) is not None and hid is not None:
        h = bb.engram.forward(h, hid[bb.engram.hash_index])
    h, pm = bb.forward(h, 0, pm)

b = e.block(2)
apre, apost, acomb = b.hc_mixes(h, b.hc_attn_fn, b.hc_attn_scale, b.hc_attn_base)
h_n = rms_norm(b.hc_pre(h, pm), b.attn_norm, e.cfg.rms_norm_eps)
ya = b.attn.forward(h_n, 0)
x1 = b.hc_post(ya, h, apost, acomb)
fpre, fpost, fcomb = b.hc_mixes(x1, b.hc_ffn_fn, b.hc_ffn_scale, b.hc_ffn_base)
h_n2 = rms_norm(b.hc_pre(x1, apre), b.ffn_norm, e.cfg.rms_norm_eps)
print("h_n2.shape=%s" % (tuple(h_n2.shape),), flush=True)
out = moe.forward(h_n2)
print("out.shape=%s absmax=%.1f" % (tuple(out.shape), float(out.float().abs().max())), flush=True)
print("_gpu_y_buf.shape=%s" % (tuple(moe._gpu_y_buf.shape),), flush=True)
