"""每次运行只建一个引擎（同进程反复建引擎会 OOM，那是探针的问题）。"""
import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
M = r"E:\models\DeepSeek-V4.1-Flash-1017"

n = int(sys.argv[1])
chunk = int(sys.argv[2]) if len(sys.argv) > 2 else 4
tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)
e = Engine(M, tier=OptLevel.AGGRESSIVE, device="cuda", layers=list(range(n)), tokenizer=tok)
ids = torch.randint(100, 1000, (1, chunk), dtype=torch.long, device="cuda")
try:
    lg = e.forward(ids, 0, chunk=chunk)
    torch.cuda.synchronize()
    ok = bool(torch.isfinite(lg).all())
    am = float(lg.float().abs().max())
    f, t = torch.cuda.mem_get_info()
    print("RESULT n=%d chunk=%d finite=%s absmax=%.2f free=%.0fMB" % (
          n, chunk, ok, am, f / 2**20), flush=True)
except Exception as ex:
    print("RESULT n=%d chunk=%d FAIL %s" % (n, chunk, str(ex)[:70]), flush=True)
