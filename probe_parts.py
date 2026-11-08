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
pos = torch.tensor([1, 2], dtype=torch.long, device="cuda")

def reset_all():
    for sub in (b.attn, getattr(b.attn, "compressor", None), getattr(b.attn, "indexer", None)):
        if sub is not None and hasattr(sub, "reset"):
            sub.reset()

def run_attn(n):
    reset_all()
    pm = torch.zeros(n, 1, HC, dtype=torch.float32, device="cuda")
    pm[:, :, 0] = 1.0
    h = b.hc_pre(x[:n], pm)
    from tinyinfer.nn import rms_norm
    h = rms_norm(h, b.attn_norm, e.cfg.rms_norm_eps)
    o = b.attn.forward(h, 0, positions=pos[:n])
    return o.float()

a1 = run_attn(1)
a2 = run_attn(2)
print("attn shapes", tuple(a1.shape), tuple(a2.shape), flush=True)
print("attn row0 maxdiff = %.6f" % float((a2[0] - a1[0]).abs().max()), flush=True)

def run_ffn(n):
    pm = torch.zeros(n, 1, HC, dtype=torch.float32, device="cuda")
    pm[:, :, 0] = 1.0
    h = b.hc_pre(x[:n], pm)
    from tinyinfer.nn import rms_norm
    h = rms_norm(h, b.ffn_norm, e.cfg.rms_norm_eps)
    return b.ffn.forward(h).float()

f1 = run_ffn(1)
f2 = run_ffn(2)
print("ffn row0 maxdiff = %.6f" % float((f2[0] - f1[0]).abs().max()), flush=True)
