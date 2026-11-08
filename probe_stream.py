"""prefill+reset 之后：哪些层被搬到 CPU，权重是否还在。"""
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

def state(tag):
    st = sorted(e._stream_layers())
    devs = []
    for i in range(4):
        b = e.block(i)
        devs.append(str(getattr(b, "_device", "?")))
    print("%-18s stream=%s  block_dev=%s  gpu_layers=%d" % (tag, st, devs, e.gpu_layers), flush=True)

state("initial")
e.reset()
state("after reset")
e._forward_chunk(L, 0)
torch.cuda.synchronize()
state("after prefill")
e.reset()
state("after reset#2")
lg = e._forward_chunk(S, 0)
torch.cuda.synchronize()
state("after short")
print("short finite=%s" % bool(torch.isfinite(lg).all()), flush=True)
