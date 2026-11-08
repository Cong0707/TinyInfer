"""8GB 能跑几层：一个进程一个引擎（上一轮总结的纪律）。"""
import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
M = r"E:\models\DeepSeek-V4.1-Flash-1017"

n = int(sys.argv[1])
tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)
t = "1+1 等于几？只回答数字。"
ids = torch.tensor([tok.encode(t, add_special_tokens=False)], device="cuda")
e = Engine(M, tier=OptLevel.AGGRESSIVE, device="cuda", layers=list(range(n)), tokenizer=tok)
e.reset()
lg = e.forward(ids, 0)
torch.cuda.synchronize()
f = lg.float()
ok = bool(torch.isfinite(f).all())
f2, t2 = torch.cuda.mem_get_info()
print("RESULT n=%2d finite=%-5s absmax=%8.3f free=%5.0fMB chunk=%d" % (
      n, ok, float(f.abs().max()) if ok else float("nan"),
      f2 / 2**20, e.prefill_chunk), flush=True)
