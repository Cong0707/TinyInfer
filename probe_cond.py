import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
M = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)
e = Engine(M, tier=OptLevel.AGGRESSIVE, device="cuda", layers=[0,1,2,3], tokenizer=tok)
L = torch.tensor([tok.encode("1+1 等于几？只回答数字。", add_special_tokens=False)], device="cuda")
S = torch.tensor([tok.encode("你好", add_special_tokens=False)], device="cuda")

def run(tag, do_prefill, do_reset):
    e.reset()
    if do_prefill:
        e._forward_chunk(L, 0)
        torch.cuda.synchronize()
    if do_reset:
        e.reset()
    lg = e._forward_chunk(S, 0)
    torch.cuda.synchronize()
    ok = bool(torch.isfinite(lg).all())
    print("%-22s finite=%s" % (tag, ok), flush=True)
    return ok

run("prefill+reset", True, True)
run("prefill only", True, False)
run("reset only", False, True)
run("neither", False, False)
