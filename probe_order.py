import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer

MODEL = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

# 关键：同一引擎、同一份 h，只改 batch 维，先测双行再测单行，
# 排除「第一次跑时 wo_a 缓存还没建好」这种顺序效应。
e = Engine(MODEL, tier=OptLevel.AGGRESSIVE, device="cuda", layers=[0], tokenizer=tok)
b = e._blocks[0]
HC, D = e.cfg.hc_mult, e.cfg.dim
torch.manual_seed(0)
x1 = torch.randn(1, 1, HC, D, dtype=torch.bfloat16, device="cuda") * 0.02
from tinyinfer.nn import rms_norm
pm1 = torch.zeros(1, 1, HC, dtype=torch.float32, device="cuda"); pm1[:, :, 0] = 1.0
h1 = rms_norm(b.hc_pre(x1, pm1), b.attn_norm, e.cfg.rms_norm_eps)

def fresh():
    b.attn.reset(); b.attn._kv_buf = None; b.attn._decode_idxs = None

pos1 = torch.tensor([1], device="cuda")
pos2 = torch.tensor([1, 1], device="cuda")

# 预热两次让缓存稳定
fresh(); b.attn.forward(h1, 0, positions=pos1)
fresh(); b.attn.forward(torch.cat([h1,h1],0), 0, positions=pos2)

# 正式对比：先双行
fresh(); o2 = b.attn.forward(torch.cat([h1,h1],0), 0, positions=pos2).float()
fresh(); o1 = b.attn.forward(h1, 0, positions=pos1).float()
print("row0 vs single (db-first) = %.6f" % float((o2[0:1] - o1).abs().max()), flush=True)

# 再反过来：先单行
fresh(); o1b = b.attn.forward(h1, 0, positions=pos1).float()
fresh(); o2b = b.attn.forward(torch.cat([h1,h1],0), 0, positions=pos2).float()
print("row0 vs single (sg-first) = %.6f" % float((o2b[0:1] - o1b).abs().max()), flush=True)
print("single stable? = %.6f" % float((o1 - o1b).abs().max()), flush=True)
