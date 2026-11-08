"""Expert 对象的实际结构。"""
import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
M = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)
e = Engine(M, tier=OptLevel.AGGRESSIVE, device="cuda", layers=[0,1,2], tokenizer=tok)
moe = e.block(0).ffn
st = moe._expert_store
rec = st.fetch(0, [3], cache=True)[0]
ex = moe._make_expert(rec)
print("type:", type(ex).__name__)
for k, v in vars(ex).items():
    if torch.is_tensor(v):
        f = v.float() if v.dtype not in (torch.uint8,) else None
        ok = bool(torch.isfinite(f).all()) if f is not None else "n/a"
        print("  %-12s %-22s %s finite=%s" % (k, str(v.dtype), tuple(v.shape)[:3], ok))
    else:
        print("  %-12s %s" % (k, type(v).__name__))
