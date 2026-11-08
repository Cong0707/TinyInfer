"""初始化后显存去哪了：逐项对账。"""
import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
GB = 2**30

def rpt(tag):
    a = torch.cuda.memory_allocated() / GB
    r = torch.cuda.memory_reserved() / GB
    f, t = torch.cuda.mem_get_info()
    print("%-26s alloc=%.2fGB reserved=%.2fGB free=%.2fGB" % (tag, a, r, f / GB), flush=True)

rpt("before")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
M = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)
rpt("tokenizer")
e = Engine(M, tier=OptLevel.AGGRESSIVE, device="cuda", layers=list(range(40)), tokenizer=tok)
rpt("after engine")

# 逐项
try:
    from tinyinfer.deepseek.gemv_fp4_marlin import cache_capacity
    print("marlin cache:", cache_capacity() if callable(cache_capacity) else "?", flush=True)
except Exception:
    pass
sl = e.expert_store.slots
print("slots=%s blob=%s -> %.2fGB" % (
    getattr(sl, "slots", "?"), getattr(sl, "blob_bytes", "?"),
    (getattr(sl, "slots", 0) or 0) * (getattr(sl, "blob_bytes", 0) or 0) / GB), flush=True)
print("gpu_layers =", e.gpu_layers, "prefill_chunk =", e.prefill_chunk, flush=True)
