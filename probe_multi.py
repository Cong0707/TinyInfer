"""2048 token：多个不同序列，看 NaN 是否与内容相关。"""
import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
M = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)
e = Engine(M, tier=OptLevel.AGGRESSIVE, device="cuda", layers=[0,1,2,3], tokenizer=tok)

sents = ["你好今天天气不错", "1+1 等于几？只回答数字。",
         "人工智能是计算机科学的一个分支，它企图了解智能的实质。",
         "The quick brown fox jumps over the lazy dog. "]
case = int(sys.argv[1])
base = tok.encode(sents[case % len(sents)], add_special_tokens=False)
rep = 200 + 200 * case
ids = torch.tensor([(base * rep)[:2048]], device="cuda")
e.reset()
try:
    lg = e._forward_chunk(ids, 0)
    torch.cuda.synchronize()
    f = lg.float()
    ok = bool(torch.isfinite(f).all())
    print("RESULT case=%d base_len=%d rep=%d finite=%s absmax=%s" % (
          case, len(base), rep, ok,
          ("%.2f" % float(f.abs().max())) if ok else "nan"), flush=True)
except Exception as ex:
    print("RESULT case=%d FAIL %s" % (case, str(ex)[:80]), flush=True)
