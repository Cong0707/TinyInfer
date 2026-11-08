"""reset 之后 attention 的 fp8 权重反量化缓存是否坏掉。"""
import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
M = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)
t = "1+1 等于几？只回答数字。请仔细思考后再回答。"
ids = torch.tensor([tok.encode(t, add_special_tokens=False)], device="cuda")
e = Engine(M, tier=OptLevel.AGGRESSIVE, device="cuda", layers=[0,1,2,3,4], tokenizer=tok)

def chk(tag):
    b = e.block(0)
    for name in ("wq_a", "wq_b", "wkv", "wo_a", "wo_b"):
        w = getattr(b.attn, name, None)
        if w is None:
            continue
        wt = getattr(w, "weight", w if torch.is_tensor(w) else None)
        if torch.is_tensor(wt):
            f = wt.float() if wt.dtype != torch.uint8 else None
            if f is not None:
                print("%-14s %-6s %-20s finite=%s absmax=%.3f" % (
                      tag, name, str(wt.dtype).replace("torch.",""),
                      bool(torch.isfinite(f).all()), float(f.abs().max())), flush=True)
    c = getattr(b.attn, "_bf16_cache", "ABSENT")
    print("%-14s _bf16_cache=%s" % (tag, type(c).__name__), flush=True)

e.reset()
chk("after reset#1")
e._forward_chunk(ids, 0)
torch.cuda.synchronize()
chk("after prefill")
e.reset()
chk("after reset#2")
