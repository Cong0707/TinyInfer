"""hc_pre 的输出分布：是少数异常值主导范数吗。"""
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
h = e.embed[ids.cpu()].to(e.device) if e.embed_on_cpu else e.embed[ids]
h = h.unsqueeze(2).repeat(1, 1, e.cfg.hc_mult, 1)
pm = torch.zeros(h.size(0), h.size(1), e.cfg.hc_mult, dtype=torch.float32, device=h.device)
pm[:, :, 0] = 1.0

def dist(tag, x):
    f = x.float().flatten()
    q = torch.quantile(f.abs()[torch.isfinite(f)], torch.tensor([0.5, 0.99, 0.9999], device=f.device))
    print("  %-16s absmax=%9.3f p50=%9.6f p99=%9.3f p9999=%9.3f mean=%9.4f" % (
          tag, float(f.abs().max()), float(q[0]), float(q[1]), float(q[2]), float(f.mean())))

for lid in (0, 1):
    b = e.block(lid)
    h, pm = b.forward(h, 0, pm)
b = e.block(2)
dist("L2 input", h)
hp = b.hc_pre(h, pm)
dist("hc_pre out", hp)

# 逐分量看 hc_pre 的输入 output 是谁
print("  hc_pre 输入列数:", h.shape, "pm:", pm.shape)
for c in range(pm.shape[-1]):
    dist("pm[...,%d]" % c, pm[..., c])
