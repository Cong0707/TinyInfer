"""多行 prefill 与逐 token 的 last-token logits 必须一致。"""
import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
M = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)
e = Engine(M, tier=OptLevel.AGGRESSIVE, device="cuda", layers=[0,1,2,3], tokenizer=tok)
base = tok.encode("1+1 等于几？只回答数字。请仔细思考。", add_special_tokens=False)

for N in (8, 16):
    ids = torch.tensor([base[:N]], device="cuda")
    e.reset()
    a = e._forward_chunk(ids, 0)[0, -1].float()
    torch.cuda.synchronize()
    e.reset()
    for p in range(N):
        b = e._forward_chunk(ids[:, p:p+1], p)[0, 0].float()
    torch.cuda.synchronize()
    if not (torch.isfinite(a).all() and torch.isfinite(b).all()):
        print("N=%2d  NaN in one path (a_ok=%s b_ok=%s)" % (
              N, bool(torch.isfinite(a).all()), bool(torch.isfinite(b).all())), flush=True)
        continue
    d = float((a - b).abs().max())
    print("N=%2d  batch vs token: maxdiff=%.6f  absmax=%.2f  argmax %d vs %d" % (
          N, d, float(a.abs().max()), int(a.argmax()), int(b.argmax())), flush=True)
