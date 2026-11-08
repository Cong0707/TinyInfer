"""L0 的 MoE：batch(8行) vs 逐 token(1行) 在同一位置。"""
import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
from tinyinfer.nn import rms_norm
M = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)
e = Engine(M, tier=OptLevel.AGGRESSIVE, device="cuda", layers=[0], tokenizer=tok)
base = tok.encode("1+1 等于几？只回答数字。请仔细思考。", add_special_tokens=False)
N = 8
ids = torch.tensor([base[:N]], device="cuda")
b = e.block(0)
moe = b.ffn

h = e.embed[ids.cpu()].to(e.device) if e.embed_on_cpu else e.embed[ids]
h = h.unsqueeze(2).repeat(1, 1, e.cfg.hc_mult, 1)

def ffn_in(seg, pos):
    n = seg.shape[1]
    pm = torch.zeros(1, n, e.cfg.hc_mult, dtype=torch.float32, device=h.device)
    pm[:, :, 0] = 1.0
    hn = rms_norm(b.hc_pre(seg, pm), b.attn_norm, e.cfg.rms_norm_eps)
    ya = b.attn.forward(hn, pos, positions=torch.arange(pos, pos+n, device="cuda").unsqueeze(0))
    x1 = b.hc_post(ya, seg, *b.hc_mixes(seg.float(), b.hc_attn_fn, b.hc_attn_scale, b.hc_attn_base)[1:])
    fp, fq, fc = b.hc_mixes(x1.float(), b.hc_ffn_fn, b.hc_ffn_scale, b.hc_ffn_base)
    return rms_norm(b.hc_pre(x1, b.hc_mixes(seg.float(), b.hc_attn_fn, b.hc_attn_scale, b.hc_attn_base)[0]), b.ffn_norm, e.cfg.rms_norm_eps)

# batch
e.reset(); b.attn._kv_buf = None
hn_b = ffn_in(h, 0)
ob = moe.forward(hn_b)
torch.cuda.synchronize()

# token 到第 7 个
e.reset(); b.attn._kv_buf = None; b.attn.reset()
last = None
for p in range(N):
    hn_p = ffn_in(h[:, p:p+1], p)
    last = moe.forward(hn_p)
torch.cuda.synchronize()

bb = ob.float()[0, N-1]; tt = last.float()[0, 0]
print("L0 moe: batch absmax=%.4f  token absmax=%.4f  maxdiff=%.6f" % (
      float(bb.abs().max()), float(tt.abs().max()), float((bb-tt).abs().max())), flush=True)
