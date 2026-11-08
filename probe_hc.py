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
x1 = torch.randn(1, 1, HC, D, dtype=torch.bfloat16, device="cuda") * 0.02
x2 = torch.cat([x1, x1], 0)

pm1 = torch.zeros(1, 1, HC, dtype=torch.float32, device="cuda"); pm1[:, :, 0] = 1.0
pm2 = torch.cat([pm1, pm1], 0)

h1 = b.hc_pre(x1, pm1).float()
h2 = b.hc_pre(x2, pm2).float()
print("hc_pre   row0 vs single = %.6f" % float((h2[0:1] - h1).abs().max()), flush=True)
print("hc_pre   rows identical = %.6f" % float((h2[0] - h2[1]).abs().max()), flush=True)

a1 = b.hc_mixes(x1, b.hc_attn_fn, b.hc_attn_scale, b.hc_attn_base)
a2 = b.hc_mixes(x2, b.hc_attn_fn, b.hc_attn_scale, b.hc_attn_base)
for i, nm in enumerate(("pre", "post", "comb")):
    print("hc_mixes %-4s row0 vs single = %.6f" % (nm, float((a2[i][0:1].float() - a1[i].float()).abs().max())), flush=True)
    print("hc_mixes %-4s rows identical = %.6f" % (nm, float((a2[i][0].float() - a2[i][1].float()).abs().max())), flush=True)
