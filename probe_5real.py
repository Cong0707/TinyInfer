"""5 层 vs 3 层：consumer 层读压缩 KV 是否越界。"""
import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
M = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)
n = int(sys.argv[1])
ids = torch.tensor([tok.encode("1+1 等于几？只回答数字。", add_special_tokens=False)], device="cuda")
e = Engine(M, tier=OptLevel.AGGRESSIVE, device="cuda", layers=list(range(n)), tokenizer=tok)
e.reset()
e._dump_h = []
lg = e._forward_chunk(ids, 0)
torch.cuda.synchronize()
print("=== n=%d ===" % n, flush=True)
for i, hin, hout in e._dump_h:
    ok_i = bool(torch.isfinite(hin).all()); ok_o = bool(torch.isfinite(hout).all())
    print("  L%d in=%-5s %9.3f | out=%-5s %9.3f" % (
          i, ok_i, float(hin.float().abs().max()),
          ok_o, float(hout.float().abs().max()) if ok_o else float("nan")), flush=True)
f = lg.float()
print("  logits finite=%s absmax=%.3f" % (bool(torch.isfinite(f).all()),
      float(f.abs().max()) if torch.isfinite(f).all() else float("nan")), flush=True)
