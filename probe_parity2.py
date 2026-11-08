"""逐位置对照：整批 prefill vs 逐 token（每个位置都存下来）。"""
import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
M = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)
t = "1+1 等于几？只回答数字。请仔细思考后再回答。"
ids = torch.tensor([tok.encode(t, add_special_tokens=False)], device="cuda")
N = ids.shape[1]
e = Engine(M, tier=OptLevel.AGGRESSIVE, device="cuda", layers=[0,1,2,3,4], tokenizer=tok)

e.reset()
la = e._forward_chunk(ids, 0)
torch.cuda.synchronize()

e.reset()
per = []
for p in range(N):
    lg = e._forward_chunk(ids[:, p:p+1], p)
    torch.cuda.synchronize()
    per.append(lg[0, 0].float().clone())

fa = la.float()
print("tokens=%d  batch_finite=%s" % (N, bool(torch.isfinite(fa).all())), flush=True)
worst = 0.0; worst_p = -1
for p in range(N):
    a = fa[0, p]; b = per[p]
    if not torch.isfinite(b).all():
        print("  pos %2d: token path NaN" % p, flush=True)
        continue
    d = float((a - b).abs().max())
    rel = d / max(1e-9, float(a.abs().max()))
    if d > worst:
        worst = d; worst_p = p
print("worst maxdiff = %.6f at pos %d" % (worst, worst_p), flush=True)
print("argmax batch=%s token=%s" % (
      [int(fa[0, p].argmax()) for p in (0, N//2, N-1)],
      [int(per[p].argmax()) for p in (0, N//2, N-1)]), flush=True)
