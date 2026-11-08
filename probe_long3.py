"""继续加长：4096/8192 token 是否还有障碍。"""
import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
M = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)
e = Engine(M, tier=OptLevel.AGGRESSIVE, device="cuda", layers=[0,1,2,3], tokenizer=tok)
base = tok.encode("你好今天天气不错", add_special_tokens=False)
for N in (4096, 8192):
    ids = torch.tensor([(base*1500)[:N]], device="cuda")
    e.reset()
    try:
        lg = e._forward_chunk(ids, 0)
        torch.cuda.synchronize()
        f = lg.float()
        ok = bool(torch.isfinite(f).all())
        print("N=%5d finite=%-5s absmax=%s" % (N, ok,
              ("%.2f" % float(f.abs().max())) if ok else "nan"), flush=True)
    except Exception as ex:
        print("N=%5d FAIL %s" % (N, str(ex)[:90]), flush=True)
