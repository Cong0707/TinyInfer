"""engram 是不是 NaN 的来源。"""
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

# layer 0
b0 = e.block(0)
h, pm = b0.forward(h, 0, pm)
print("after L0 finite=%s absmax=%.3f" % (bool(torch.isfinite(h).all()), float(h.float().abs().max())))

# layer 1: 先看 engram 的输出
b1 = e.block(1)
eng = getattr(b1, "engram", None)
print("layer1 has engram:", eng is not None)
if eng is not None:
    hidx = getattr(eng, "hash_index", 0)
    print("hash_index =", hidx)
    try:
        he = eng.forward(h, None)
        print("engram(h, None) finite=%s absmax=%.3f" % (
              bool(torch.isfinite(he).all()), float(he.float().abs().max())))
    except Exception as ex:
        print("engram(h, None) EXC", str(ex)[:100])
