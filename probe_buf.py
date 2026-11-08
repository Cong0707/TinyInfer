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
x = torch.randn(1, 1, HC, D, dtype=torch.bfloat16, device="cuda") * 0.02
from tinyinfer.nn import rms_norm

b.attn.reset()
pm = torch.zeros(1, 1, HC, dtype=torch.float32, device="cuda"); pm[:, :, 0] = 1.0
h1 = rms_norm(b.hc_pre(x, pm), b.attn_norm, e.cfg.rms_norm_eps)
h2 = torch.cat([h1, h1], 0)

b.attn.reset()
# 清掉可能残留的 buf
if hasattr(b.attn, "_kv_buf") and b.attn._kv_buf is not None:
    b.attn._kv_buf = None
o2 = b.attn.forward(h2, 0, positions=torch.tensor([1, 1], device="cuda")).float()
print("clean buf: two identical rows maxdiff = %.6f" % float((o2[0] - o2[1]).abs().max()), flush=True)
buf = b.attn._kv_buf
print("buf rows maxdiff = %.6f" % float((buf[0].float() - buf[1].float()).abs().max()), flush=True)
print("buf[0] nonzero slots =", int((buf[0].float().abs().sum(-1) > 0).sum()), flush=True)
