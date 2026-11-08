"""reset 是否把 shared_attn.compress_kv 置为 None，导致后续 NaN。"""
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

def show(tag):
    ck = getattr(SA, "compress_kv", None)
    print("%-14s compress_kv=%s" % (tag,
          None if ck is None else (tuple(ck.shape), bool(torch.isfinite(ck.float()).all()))), flush=True)

e.reset()
show("after reset#1")
la = e._forward_chunk(ids, 0)
torch.cuda.synchronize()
show("after prefill")
e.reset()
show("after reset#2")
try:
    lg = e._forward_chunk(ids[:, :1], 0)
    torch.cuda.synchronize()
    f = lg.float()
    print("decode finite=%s" % bool(torch.isfinite(f).all()), flush=True)
    show("after decode")
except Exception as ex:
    print("decode EXC %s" % str(ex)[:80], flush=True)
