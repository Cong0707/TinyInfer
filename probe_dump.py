"""在 _forward_chunk 的层循环里打点，看 NaN 首次出现。"""
import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
M = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)
t = "1+1 等于几？只回答数字。"
ids = torch.tensor([tok.encode(t, add_special_tokens=False)], device="cuda")
e = Engine(M, tier=OptLevel.AGGRESSIVE, device="cuda", layers=[0,1,2,3,4], tokenizer=tok)
e.reset()

# 用 _dump_h 钩子，它在 _forward_chunk 里逐层记录 (i, h_in, h_out)
dumps = []
e._dump_h = dumps
lg = e.forward(ids, 0)
torch.cuda.synchronize()
print("logits finite=%s" % bool(torch.isfinite(lg).all()))
for i, hin, hout in dumps:
    print("layer %d: in finite=%-5s absmax=%9.3f | out finite=%-5s absmax=%9.3f" % (
          i, bool(torch.isfinite(hin).all()), float(hin.float().abs().max()),
          bool(torch.isfinite(hout).all()),
          float(hout.float().abs().max()) if torch.isfinite(hout).all() else float("nan")))
