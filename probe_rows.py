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

# 两行完全相同的数据，位置也相同 -> 两行输出必须一致
reset()
h = prep(2).clone()
h[1] = h[0]                     # 第 1 行 = 第 0 行
pos = torch.tensor([1, 1], device="cuda")
o = b.attn.forward(h, 0, positions=pos).float()
print("rows identical? maxdiff = %.6f" % float((o[0] - o[1]).abs().max()), flush=True)

# 单行对照
reset()
os1 = b.attn.forward(h[0:1], 0, positions=torch.tensor([1], device="cuda")).float()
print("row0 vs single = %.6f" % float((o[0] - os1[0]).abs().max()), flush=True)
