"""同一行、同一路由，batch(n_exp=48) vs 单行(n_exp=6) 的 MoE 输出。"""
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
b = e.block(0); moe = b.ffn
h = e.embed[ids.cpu()].to(e.device) if e.embed_on_cpu else e.embed[ids]
h = h.unsqueeze(2).repeat(1, 1, e.cfg.hc_mult, 1)
pm = torch.zeros(1, N, e.cfg.hc_mult, dtype=torch.float32, device=h.device)
pm[:, :, 0] = 1.0
hn = rms_norm(b.hc_pre(h, pm), b.attn_norm, e.cfg.rms_norm_eps)
row = N - 1

# A: 8 行一次
e.reset()
out_b = moe.forward(hn)
torch.cuda.synchronize()
a = out_b.float()[0, row]

# B: 只送那一行
e.reset()
out_s = moe.forward(hn[:, row:row+1])
torch.cuda.synchronize()
s = out_s.float()[0, 0]

print("batch absmax=%.4f  single absmax=%.4f  maxdiff=%.6f" % (
      float(a.abs().max()), float(s.abs().max()), float((a-s).abs().max())), flush=True)
print("ratio diff/absmax = %.3f" % (float((a-s).abs().max()) / max(1e-9, float(a.abs().max()))), flush=True)
# 逐元素看最大的几处
d = (a - s).abs()
idx = d.topk(5).indices.tolist()
print("top diffs at dims:", idx, flush=True)
for i in idx[:3]:
    print("  dim %4d: batch=%9.5f single=%9.5f diff=%9.5f" % (
          i, float(a[i]), float(s[i]), float(d[i])), flush=True)
