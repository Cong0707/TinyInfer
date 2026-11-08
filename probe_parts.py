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

def fresh():
    b.attn.reset()
    b.attn._kv_buf = None

def prep(n):
    pm = torch.zeros(n, 1, HC, dtype=torch.float32, device="cuda")
    pm[:, :, 0] = 1.0
    return rms_norm(b.hc_pre(x[:n], pm), b.attn_norm, e.cfg.rms_norm_eps)

# 单行，position 1
fresh(); h1 = prep(1)
o1 = b.attn.forward(h1, 0, positions=torch.tensor([1], device="cuda")).float()

# 两行，位置 [1, 2]：第 0 行应等于 o1
fresh(); h2 = prep(2)
o2 = b.attn.forward(h2, 0, positions=torch.tensor([1, 2], device="cuda")).float()
print("row0 vs single  = %.6f" % float((o2[0:1] - o1).abs().max()), flush=True)

# 两行，位置 [1, 1]
fresh(); h3 = prep(2)
o3 = b.attn.forward(h3, 0, positions=torch.tensor([1, 1], device="cuda")).float()
print("row0 pos[1,1]   = %.6f" % float((o3[0:1] - o1).abs().max()), flush=True)

# 两行，位置都是 1，但第 1 行数据与第 0 行相同 -> 两行输出必须一致
fresh()
h4 = h3.clone(); h4[1] = h4[0]
o4 = b.attn.forward(h4, 0, positions=torch.tensor([1, 1], device="cuda")).float()
print("identical rows  = %.6f" % float((o4[0] - o4[1]).abs().max()), flush=True)
