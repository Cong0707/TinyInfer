"""同一个 hn2，直接调 moe.forward vs 走 _forward_chunk。"""
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
b = e.block(0)
h = e.embed[S.cpu()].to(e.device) if e.embed_on_cpu else e.embed[S]
h = h.unsqueeze(2).repeat(1, 1, e.cfg.hc_mult, 1)
pm = torch.zeros(1, 1, e.cfg.hc_mult, dtype=torch.float32, device=h.device)
pm[:, :, 0] = 1.0
hid = e.hashes(S.reshape(-1).tolist(), 0) if e.hashes is not None else None
if getattr(b, "engram", None) is not None and hid is not None:
    h = b.engram.forward(h, hid[b.engram.hash_index])
apre, apost, acomb = b.hc_mixes(h.float(), b.hc_attn_fn, b.hc_attn_scale, b.hc_attn_base)
hn = rms_norm(b.hc_pre(h, pm), b.attn_norm, e.cfg.rms_norm_eps)
ya = b.attn.forward(hn, 0)
x1 = b.hc_post(ya, h, apost, acomb)
fpre, fpost, fcomb = b.hc_mixes(x1.float(), b.hc_ffn_fn, b.hc_ffn_scale, b.hc_ffn_base)
hn2 = rms_norm(b.hc_pre(x1, apre), b.ffn_norm, e.cfg.rms_norm_eps)
torch.cuda.synchronize()
print("hn2 finite=%s absmax=%.4f" % (bool(torch.isfinite(hn2).all()),
      float(hn2.float().abs().max())), flush=True)

out = b.ffn.forward(hn2)
torch.cuda.synchronize()
print("moe.forward out finite=%s absmax=%s" % (bool(torch.isfinite(out).all()),
      ("%.4f" % float(out.float().abs().max())) if torch.isfinite(out).all() else "nan"), flush=True)
print("_gpu_fail =", getattr(b.ffn, "_gpu_fail", None), flush=True)
