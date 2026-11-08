"""行数变化是否必然 NaN（不管有没有 reset）。"""
import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
M = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)
e = Engine(M, tier=OptLevel.AGGRESSIVE, device="cuda", layers=[0,1,2,3], tokenizer=tok)
base = tok.encode("1+1 等于几？只回答数字。", add_special_tokens=False)

def run(a, b, do_reset):
    e.reset()
    lg1 = e._forward_chunk(torch.tensor([base[:a]], device="cuda"), 0)
    torch.cuda.synchronize()
    if do_reset:
        e.reset()
    lg2 = e._forward_chunk(torch.tensor([base[:b]], device="cuda"), 0)
    torch.cuda.synchronize()
    ok1 = bool(torch.isfinite(lg1).all()); ok2 = bool(torch.isfinite(lg2).all())
    print("  %2d -> %2d  reset=%-5s  first=%-5s second=%-5s" % (a, b, do_reset, ok1, ok2), flush=True)

run(11, 11, False)
run(11, 1, False)
run(1, 11, False)
run(1, 1, False)
run(11, 4, False)
run(4, 11, False)
