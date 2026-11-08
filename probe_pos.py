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

def reset():
    b.attn.reset()

def prep(n):
    pm = torch.zeros(n, 1, HC, dtype=torch.float32, device="cuda")
    pm[:, :, 0] = 1.0
    h = b.hc_pre(x[:n], pm)
    return rms_norm(h, b.attn_norm, e.cfg.rms_norm_eps)

# 单行、位置 1
reset(); h1 = prep(1)
o1 = b.attn.forward(h1, 0, positions=torch.tensor([1], device="cuda")).float()

# 两行、位置 [1, 2]，取第 0 行
reset(); h2 = prep(2)
o2 = b.attn.forward(h2, 0, positions=torch.tensor([1, 2], device="cuda")).float()
print("row0 maxdiff (pos 1 vs [1,2]) = %.6f" % float((o2[0:1] - o1).abs().max()), flush=True)

# 两行、位置都是 1（同位置）
reset(); h3 = prep(2)
o3 = b.attn.forward(h3, 0, positions=torch.tensor([1, 1], device="cuda")).float()
print("row0 maxdiff (pos 1 vs [1,1]) = %.6f" % float((o3[0:1] - o1).abs().max()), flush=True)
