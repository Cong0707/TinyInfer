"""engram 对 h 的影响（修掉 truth-value 判空）。"""
import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
M = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)
t = "1+1 等于几？只回答数字。"
ids = torch.tensor([tok.encode(t, add_special_tokens=False)], device="cuda")
e = Engine(M, tier=OptLevel.AGGRESSIVE, device="cuda", layers=[0,1,2], tokenizer=tok)
e.reset()
h = e.embed[ids.cpu()].to(e.device) if e.embed_on_cpu else e.embed[ids]
h = h.unsqueeze(2).repeat(1, 1, e.cfg.hc_mult, 1)
pm = torch.zeros(h.size(0), h.size(1), e.cfg.hc_mult, dtype=torch.float32, device=h.device)
pm[:, :, 0] = 1.0

def st(tag, x):
    f = x.float()
    print("  %-24s finite=%-5s absmax=%10.3f" % (tag, bool(torch.isfinite(f).all()), float(f.abs().max())))

b0 = e.block(0)
h, pm = b0.forward(h, 0, pm)
st("before L1", h)

b1 = e.block(1)
hid = e.hashes(ids.reshape(-1).tolist(), 0)
print("hash_ids len:", len(hid), flush=True)
hi = b1.engram.hash_index
print("engram hash_index:", hi, flush=True)
prev = h.clone()
h2 = b1.engram.forward(h, hid[hi])
st("after engram", h2)
d = (h2.float() - prev.float()).abs()
print("  delta absmax=%.4f mean=%.6f" % (float(d.max()), float(d.mean())), flush=True)
st("h2 (returned)", h2)
