import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
import tinyinfer.deepseek.attention as AT

MODEL = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
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

# 真实调用约定：start_pos 与 positions 一致
P = 1
fresh()
o1 = b.attn.forward(h1, P, positions=torch.tensor([P], device="cuda")).float()
fresh()
o2 = b.attn.forward(torch.cat([h1,h1],0), P, positions=torch.tensor([P,P], device="cuda")).float()
print("start_pos consistent: row0 vs single = %.6f" % float((o2[0:1] - o1).abs().max()), flush=True)
print("rows identical = %.6f" % float((o2[0] - o2[1]).abs().max()), flush=True)
