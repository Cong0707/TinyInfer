"""拦截 MoE 的 _gpu_only 和 _sh。"""
import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
M = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)
t = "1+1 等于几？只回答数字。"
ids = torch.tensor([tok.encode(t, add_special_tokens=False)], device="cuda")
e = Engine(M, tier=OptLevel.AGGRESSIVE, device="cuda", layers=[0,1,2], tokenizer=tok)
e.reset()

b = e.block(2)
moe = b.ffn
capt = {}
orig = moe.shared.forward
def hooked(x, *a, **k):
    out = orig(x, *a, **k)
    capt.setdefault("sh", []).append(out.detach().float().clone())
    capt.setdefault("sh_in", []).append(x.detach().float().clone())
    return out
moe.shared.forward = hooked

orig_moe_forward = moe.forward
def hooked_moe(x, *a, **k):
    out = orig_moe_forward(x, *a, **k)
    capt.setdefault("out", []).append(out.detach().float().clone())
    capt.setdefault("in", []).append(x.detach().float().clone())
    return out
moe.forward = hooked_moe

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

from tinyinfer.nn import rms_norm
apre, apost, acomb = b.hc_mixes(h, b.hc_attn_fn, b.hc_attn_scale, b.hc_attn_base)
h_n = rms_norm(b.hc_pre(h, pm), b.attn_norm, e.cfg.rms_norm_eps)
ya = b.attn.forward(h_n, 0)
x1 = b.hc_post(ya, h, apost, acomb)
fpre, fpost, fcomb = b.hc_mixes(x1, b.hc_ffn_fn, b.hc_ffn_scale, b.hc_ffn_base)
h_n2 = rms_norm(b.hc_pre(x1, apre), b.ffn_norm, e.cfg.rms_norm_eps)

out = b.ffn.forward(h_n2)
torch.cuda.synchronize()
print("MoE in   absmax=%.3f" % float(capt["in"][0].abs().max()), flush=True)
print("shared   absmax=%.3f" % float(capt["sh"][0].abs().max()), flush=True)
print("MoE out  absmax=%.3f" % float(capt["out"][0].abs().max()), flush=True)
