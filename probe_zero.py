"""短序列 NaN：chunked prefill 的根本障碍。"""
import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
M = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)
t = "1+1 等于几？只回答数字。"
ids = torch.tensor([tok.encode(t, add_special_tokens=False)], device="cuda")
e = Engine(M, tier=OptLevel.AGGRESSIVE, device="cuda", layers=[0,1,2,3], tokenizer=tok)
print("total tokens:", ids.shape[1], flush=True)
for n in (1, 2, 3, 4, 6, 8, 11):
    e.reset()
    lg = e._forward_chunk(ids[:, :n], 0)
    torch.cuda.synchronize()
    f = lg.float()
    ok = bool(torch.isfinite(f).all())
    print("  first %2d tokens: finite=%s absmax=%.2f" % (
          n, ok, float(f.abs().max()) if ok else float("nan")), flush=True)
