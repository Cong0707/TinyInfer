"""用真实 token 序列，排除「随机 id 导致激活爆炸」的可能。"""
import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
M = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)
e = Engine(M, tier=OptLevel.AGGRESSIVE, device="cuda", layers=list(range(8)), tokenizer=tok)

text = "1+1 等于几？只回答数字。"
ids = torch.tensor([tok.encode(text, add_special_tokens=False)], device="cuda")
print("real tokens:", ids.shape, flush=True)
e.reset()
lg = e.forward(ids, 0, chunk=4)
torch.cuda.synchronize()
print("full 8-layer forward: finite=%s absmax=%.2f" % (
      bool(torch.isfinite(lg).all()), float(lg.float().abs().max())), flush=True)

# 逐层
e.reset()
h = e.embed[ids.cpu()].to(e.device) if e.embed_on_cpu else e.embed[ids]
h = h.unsqueeze(2).repeat(1, 1, e.cfg.hc_mult, 1)
pm = torch.zeros(h.size(0), h.size(1), e.cfg.hc_mult, dtype=torch.float32, device=h.device)
pm[:, :, 0] = 1.0
for lid in range(8):
    b = e.block(lid)
    o = b.forward(h, 0, pm)
    h = o[0] if isinstance(o, tuple) else o
    torch.cuda.synchronize()
    ok = bool(torch.isfinite(h).all())
    print("  layer %d finite=%s absmax=%.2f" % (lid, ok, float(h.float().abs().max())), flush=True)
    if not ok:
        break
