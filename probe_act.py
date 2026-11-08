"""rows=1 时：moe_fused_act 输出到 scr 的内容是否 finite。"""
import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
from tinyinfer.nn import rms_norm
import tinyinfer.deepseek.moe as MOE
M = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)

# 拦截 BoundLaunch 的 __call__，在 act 之后检查 scr
cap = {}
from tinyinfer.kernels import native as NAT
orig_launch = NAT.BoundLaunch.__call__
def patched(self, *a, **k):
    r = orig_launch(self, *a, **k)
    if a and torch.is_tensor(a[0]) and a[0].dtype == torch.float8_e4m3fn:
        torch.cuda.synchronize()
        f = a[0].float()
        cap["act_finite"] = bool(torch.isfinite(f).all())
        cap["act_absmax"] = float(f.abs().max()) if cap["act_finite"] else None
        cap["act_numel"] = int(f.numel())
    return r
NAT.BoundLaunch.__call__ = patched

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
out = b.ffn.forward(hn2)
torch.cuda.synchronize()
print("act: finite=%s absmax=%s numel=%s" % (cap.get("act_finite"),
      cap.get("act_absmax"), cap.get("act_numel")), flush=True)
print("moe out finite=%s" % bool(torch.isfinite(out).all()), flush=True)
