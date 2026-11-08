import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer

MODEL = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
e = Engine(MODEL, tier=OptLevel.AGGRESSIVE, device="cuda", layers=[0], tokenizer=tok)
b = e._blocks[0]
HC = e.cfg.hc_mult
D = e.cfg.dim

# 直接用随机 hidden 测 block，排除 embedding 的影响
torch.manual_seed(0)
x = torch.randn(2, 1, HC, D, dtype=torch.bfloat16, device="cuda") * 0.02
pos = torch.tensor([1, 2], dtype=torch.long, device="cuda")

def run(n):
    for sub in (b.attn, getattr(b.attn, "compressor", None), getattr(b.attn, "indexer", None)):
        if sub is not None and hasattr(sub, "reset"):
            sub.reset()
    pm = torch.zeros(n, 1, HC, dtype=torch.float32, device="cuda")
    pm[:, :, 0] = 1.0
    o = b.forward(x[:n], 0, pm, positions=pos[:n])
    # Block.forward 返回 (h, pre_mix)
    h = o[0] if isinstance(o, tuple) else o
    return h.float()

o1 = run(1)
o2 = run(2)
print("shapes", tuple(o1.shape), tuple(o2.shape), flush=True)
print("row0 maxdiff = %.5f" % float((o2[0] - o1[0]).abs().max()), flush=True)
print("row0 absmax single=%.4f batch=%.4f" % (float(o1[0].abs().max()), float(o2[0].abs().max())), flush=True)
