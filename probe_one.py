"""单独跑一个长度（不先跑别的），避免 OOM 污染 context。"""
import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
M = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)
N = int(sys.argv[1])
e = Engine(M, tier=OptLevel.AGGRESSIVE, device="cuda", layers=[0,1,2,3], tokenizer=tok)
base = tok.encode("你好今天天气不错", add_special_tokens=False)
ids = torch.tensor([(base*2000)[:N]], device="cuda")
e.reset()
try:
    lg = e._forward_chunk(ids, 0)
    torch.cuda.synchronize()
    f = lg.float()
    ok = bool(torch.isfinite(f).all())
    print("RESULT N=%d finite=%s absmax=%s" % (N, ok,
          ("%.2f" % float(f.abs().max())) if ok else "nan"), flush=True)
except Exception as ex:
    print("RESULT N=%d FAIL %s" % (N, str(ex)[:100]), flush=True)
