"""第 2 层（kv_source）为什么 NaN：逐段检查。"""
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

h = e.embed[ids.cpu()].to(e.device) if e.embed_on_cpu else e.embed[ids]
h = h.unsqueeze(2).repeat(1, 1, e.cfg.hc_mult, 1)
pm = torch.zeros(h.size(0), h.size(1), e.cfg.hc_mult, dtype=torch.float32, device=h.device)
pm[:, :, 0] = 1.0

for lid in (0, 1):
    b = e.block(lid)
    o = b.forward(h, 0, pm)
    h = o[0] if isinstance(o, tuple) else o
    print("layer %d finite=%s absmax=%.3f" % (lid,
          bool(torch.isfinite(h).all()), float(h.float().abs().max())))

b = e.block(2)
a = b.attn
from tinyinfer.nn import rms_norm
h2 = rms_norm(b.hc_pre(h, pm), b.attn_norm, e.cfg.rms_norm_eps)
print("pre-attn finite=%s absmax=%.4f" % (bool(torch.isfinite(h2).all()),
      float(h2.float().abs().max())))

o = a.forward(h2, 0)
print("attn out finite=%s absmax=%.3f" % (bool(torch.isfinite(o).all()),
      float(o.float().abs().max())))

# 压缩相关状态
rt = getattr(a, "rt", None)
if rt is not None:
    for k in ("compress_kv", "index_k", "topk_idxs", "candidates"):
        v = getattr(rt, k, None)
        if torch.is_tensor(v):
            print("  rt.%-12s finite=%s absmax=%.4f" % (k,
                  bool(torch.isfinite(v.float()).all()), float(v.float().abs().max())))
