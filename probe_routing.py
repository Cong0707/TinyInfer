"""同一行在 batch 与单行路径下的路由（专家 id + 权重）是否一致。"""
import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
from tinyinfer.nn import rms_norm
M = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)
e = Engine(M, tier=OptLevel.AGGRESSIVE, device="cuda", layers=[0], tokenizer=tok)
base = tok.encode("1+1 等于几？只回答数字。请仔细思考。", add_special_tokens=False)
N = 8
ids = torch.tensor([base[:N]], device="cuda")
b = e.block(0)
moe = b.ffn

h = e.embed[ids.cpu()].to(e.device) if e.embed_on_cpu else e.embed[ids]
h = h.unsqueeze(2).repeat(1, 1, e.cfg.hc_mult, 1)
pm = torch.zeros(1, N, e.cfg.hc_mult, dtype=torch.float32, device=h.device)
pm[:, :, 0] = 1.0
hn = rms_norm(b.hc_pre(h, pm), b.attn_norm, e.cfg.rms_norm_eps)

xf = hn.view(-1, e.cfg.dim)
w_b, i_b = moe.gate.forward(xf, None)
torch.cuda.synchronize()
print("batch gate: xf=%s w=%s idx=%s" % (tuple(xf.shape), tuple(w_b.shape), tuple(i_b.shape)), flush=True)
row = N - 1
print("  row %d: idx=%s" % (row, i_b[row].tolist()), flush=True)
print("  row %d: w  =%s" % (row, [round(float(x), 6) for x in w_b[row].tolist()]), flush=True)

# 单行
x1 = hn[:, row:row+1].view(-1, e.cfg.dim)
w_s, i_s = moe.gate.forward(x1, None)
torch.cuda.synchronize()
print("single gate: idx=%s" % (i_s[0].tolist(),), flush=True)
print("             w  =%s" % ([round(float(x), 6) for x in w_s[0].tolist()],), flush=True)
print("idx same: %s   w maxdiff: %.8f" % (
      i_b[row].tolist() == i_s[0].tolist(),
      float((w_b[row].float() - w_s[0].float()).abs().max())), flush=True)
