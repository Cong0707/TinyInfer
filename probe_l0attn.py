"""L0 的 attention：batch(8行) vs 逐 token(1行) 在同一位置的输出。"""
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

# A: 一次送 8 个 token
e.reset()
h = e.embed[ids.cpu()].to(e.device) if e.embed_on_cpu else e.embed[ids]
h = h.unsqueeze(2).repeat(1, 1, e.cfg.hc_mult, 1)
pm = torch.zeros(1, N, e.cfg.hc_mult, dtype=torch.float32, device=h.device)
pm[:, :, 0] = 1.0
saved = b.attn._kv_buf
b.attn._kv_buf = None
hb = rms_norm(b.hc_pre(h, pm), b.attn_norm, e.cfg.rms_norm_eps)
ob = b.attn.forward(hb, 0, positions=torch.arange(N, device="cuda").unsqueeze(0))
torch.cuda.synchronize()

# B: 逐 token 喂到第 7 个
b.attn._kv_buf = None
b.attn.reset()
last = None
for p in range(N):
    hp = torch.zeros(1, 1, e.cfg.hc_mult, dtype=torch.float32, device=h.device)
    hp[:, :, 0] = 1.0
    hh = rms_norm(b.hc_pre(h[:, p:p+1], hp), b.attn_norm, e.cfg.rms_norm_eps)
    last = b.attn.forward(hh, p, positions=torch.tensor([[p]], device="cuda"))
torch.cuda.synchronize()

bb = ob.float()[0, N-1]; tt = last.float()[0, 0]
print("L0 attn: batch absmax=%.4f  token absmax=%.4f  maxdiff=%.6f" % (
      float(bb.abs().max()), float(tt.abs().max()), float((bb-tt).abs().max())), flush=True)
