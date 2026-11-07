
import sys, os, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
# 强制 vllm_ops 走 pyd 路径：把 native.bind 弄成永远失败
from tinyinfer import native
native.available = lambda *a, **k: False
from tinyinfer import vllm_ops
vllm_ops._BOUND.clear()
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
d = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
ids = tok("Hello, I am DeepSeek.", return_tensors="pt").input_ids[0].tolist()
eng = Engine(d, tier=OptLevel.BALANCED, device="cuda", layers=list(range(40)))
print("engine ok")
with torch.inference_mode():
    outs = list(eng.stream_generate(ids, max_new_tokens=6, temperature=0.0))
print("text:", repr(tok.decode(outs)))
