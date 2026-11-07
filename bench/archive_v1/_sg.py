
import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer

d = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
p = tok("Hello", return_tensors="pt").input_ids[0].tolist()
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device="cuda", layers=[0])
es = eng.expert_store
print("multi_gpu:", eng._multi_gpu, " store devices:", getattr(es, "devices", None))
with torch.inference_mode():
    out = list(eng.stream_generate(p, max_new_tokens=4, temperature=0.0))
print("输出:", out, "->", repr(tok.decode(out)))
print("显存峰值:", round(torch.cuda.max_memory_allocated()/2**20), "MB")
