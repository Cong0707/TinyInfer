import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer

MODEL = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
e = Engine(MODEL, tier=OptLevel.AGGRESSIVE, device="cuda", layers=[0], tokenizer=tok)
b = e._blocks[0]
HC, D = e.cfg.hc_mult, e.cfg.dim
torch.manual_seed(0)
x = torch.randn(2, 1, HC, D, dtype=torch.bfloat16, device="cuda") * 0.02
from tinyinfer.nn import rms_norm

def prep(n, pos):
    for sub in (b.attn,):
        if hasattr(sub, "reset"):
            sub.reset()
    pm = torch.zeros(n, 1, HC, dtype=torch.float32, device="cuda")
    pm[:, :, 0] = 1.0
    h = b.hc_pre(x[:n], pm)
    return rms_norm(h, b.attn_norm, e.cfg.rms_norm_eps)

# 单独跑两行，各自拿到 h；然后比较它们的 h 是否相同
h1 = prep(1, torch.tensor([1], device="cuda"))
h2 = prep(2, torch.tensor([1, 2], device="cuda"))
print("h row0 maxdiff (single vs batch prep) = %.6f" % float((h2[0:1] - h1).abs().max()), flush=True)

# 用相同的 h 分别走 attn
for sub in (b.attn,):
    if hasattr(sub, "reset"):
        sub.reset()
o1 = b.attn.forward(h1, 0, positions=torch.tensor([1], device="cuda")).float()
for sub in (b.attn,):
    if hasattr(sub, "reset"):
        sub.reset()
o2 = b.attn.forward(h2[0:1], 0, positions=torch.tensor([1], device="cuda")).float()
print("attn same-h same-pos maxdiff = %.6f" % float((o2 - o1).abs().max()), flush=True)
