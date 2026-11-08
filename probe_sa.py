"""reset 之后 shared_attn 的哪些属性被销毁。"""
import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
from tinyinfer.deepseek.compressor import shared_attn as SA
M = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)
t = "1+1 等于几？只回答数字。请仔细思考后再回答。"
ids = torch.tensor([tok.encode(t, add_special_tokens=False)], device="cuda")
e = Engine(M, tier=OptLevel.AGGRESSIVE, device="cuda", layers=[0,1,2,3,4], tokenizer=tok)
ATTRS = ("_mask_cache", "candidates_by_chunk", "candidates",
         "compress_kv", "topk_idxs", "index_k", "src_ratio")

def show(tag):
    parts = []
    for a in ATTRS:
        v = getattr(SA, a, "<absent>")
        if torch.is_tensor(v):
            parts.append("%s=T%s" % (a, tuple(v.shape)))
        elif v is None:
            parts.append("%s=None" % a)
        elif isinstance(v, (dict, list)):
            parts.append("%s=len%d" % (a, len(v)))
        elif v == "<absent>":
            parts.append("%s=ABSENT" % a)
        else:
            parts.append("%s=%r" % (a, v))
    print("%-14s %s" % (tag, " ".join(parts)), flush=True)

e.reset()
show("after reset#1")
e._forward_chunk(ids, 0)
torch.cuda.synchronize()
show("after prefill")
e.reset()
show("after reset#2")
