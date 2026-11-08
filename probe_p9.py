"""L2 的压缩写入：p9 位置是否写错。"""
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
e.reset()
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
a = b.attn
apre, apost, acomb = b.hc_mixes(h, b.hc_attn_fn, b.hc_attn_scale, b.hc_attn_base)
hn = rms_norm(b.hc_pre(h, pm), b.attn_norm, e.cfg.rms_norm_eps)
print("attn in p9 absmax=%.3f" % float(hn.float()[0, 9].abs().max()), flush=True)
ya = a.forward(hn, 0)
print("attn out p9 absmax=%.3f  (full absmax=%.3f)" % (
      float(ya.float()[0, 9].abs().max()), float(ya.float().abs().max())), flush=True)
x1 = b.hc_post(ya, h, apost, acomb)
print("after hc_post p9 absmax=%.3f" % float(x1.float()[0, 9].abs().max()), flush=True)

rt = a.rt
ck = rt.compress_kv
if ck is not None:
    c = ck.float()
    print("compress_kv shape=%s" % (tuple(ck.shape),), flush=True)
    for g in range(6):
        row = c[0, g]
        print("   group %d: absmax=%10.3f finite=%s" % (
              g, float(row.abs().max()), bool(torch.isfinite(row).all())), flush=True)
