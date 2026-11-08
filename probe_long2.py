"""修复后：不同长度的完整引擎 forward。"""
import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
M = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)
e = Engine(M, tier=OptLevel.AGGRESSIVE, device="cuda", layers=[0,1,2,3], tokenizer=tok)
base = tok.encode("你好今天天气不错", add_special_tokens=False)
for N in (128, 256, 512, 1024, 2048):
    ids = torch.tensor([(base*400)[:N]], device="cuda")
    e.reset()
    try:
        lg = e._forward_chunk(ids, 0)
        torch.cuda.synchronize()
        f = lg.float()
        ok = bool(torch.isfinite(f).all())
        print("N=%4d finite=%-5s absmax=%s" % (N, ok,
              ("%.2f" % float(f.abs().max())) if ok else "nan"), flush=True)
    except Exception as ex:
        print("N=%4d FAIL %s" % (N, str(ex)[:80]), flush=True)
