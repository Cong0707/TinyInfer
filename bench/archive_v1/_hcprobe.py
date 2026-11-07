
import torch, sys
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
d = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
ids = tok("Hello", return_tensors="pt").input_ids[0].tolist()
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device="cuda", layers=list(range(2)))
b = eng.block(0)
x = torch.randn(1, 1, b.hc_mult, 64, dtype=torch.bfloat16, device="cuda")
pre, post, comb = b.hc_mixes(x, b.hc_attn_fn, b.hc_attn_scale, b.hc_attn_base)
print("pre ", tuple(pre.shape), pre.dtype)
print("post", tuple(post.shape), post.dtype)
print("comb", tuple(comb.shape), comb.dtype)
resid = x
y = x
try:
    out = b.hc_post(y, resid, post, comb)
    print("hc_post ->", tuple(out.shape), out.dtype)
except Exception as e:
    print("hc_post FAILED:", type(e).__name__, str(e)[:150])
