
import os, sys, time, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer

d = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device="cuda", layers=list(range(40)))
prompt = tok("The history of computing begins with", return_tensors="pt").input_ids[0].tolist()
long_ids = (prompt * 40)[:160]
tag = "FUSED" if os.environ.get("TI_FUSED_MOE","1") != "0" else "SPLIT"
with torch.inference_mode():
    list(eng.stream_generate(long_ids, max_new_tokens=2, temperature=0.0))  # warm
    t0 = time.time()
    list(eng.stream_generate(long_ids, max_new_tokens=2, temperature=0.0))
    ttft = time.time() - t0
print(f"{tag}: prefill+2tok of {len(long_ids)} tokens = {ttft*1000:.0f} ms")
