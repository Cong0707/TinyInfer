"""MoE 内部哪些地方加同步能消除 NaN。"""
import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
import tinyinfer.deepseek.moe as MOE
M = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)
t = "1+1 等于几？只回答数字。请仔细思考后再回答。"
ids = torch.tensor([tok.encode(t, add_special_tokens=False)], device="cuda")
e = Engine(M, tier=OptLevel.AGGRESSIVE, device="cuda", layers=[0,1,2,3,4], tokenizer=tok)
moe = e.block(0).ffn
orig = moe._forward_gpu_only

WHERE = sys.argv[1]
def hook(x, xf, indices, weights):
    if WHERE == "before":
        torch.cuda.synchronize()
    out = orig(x, xf, indices, weights)
    if WHERE == "after":
        torch.cuda.synchronize()
    return out
moe._forward_gpu_only = hook

e.reset()
e._forward_chunk(ids, 0)
torch.cuda.synchronize()
e.reset()
lg = e._forward_chunk(ids[:, :1], 0)
torch.cuda.synchronize()
print("WHERE=%-8s decode finite=%s" % (WHERE, bool(torch.isfinite(lg).all())), flush=True)
