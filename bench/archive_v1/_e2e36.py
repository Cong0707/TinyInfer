
import sys, time, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
d = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
ids = tok("Hello, I am DeepSeek.", return_tensors="pt").input_ids[0].tolist()
eng = Engine(d, tier=OptLevel.BALANCED, device="cuda", layers=list(range(36)))
with torch.inference_mode():
    list(eng.stream_generate(ids, max_new_tokens=2, temperature=0.0))
    marks=[]
    for _ in eng.stream_generate(ids, max_new_tokens=8, temperature=0.0):
        marks.append(time.time())
outs = None
gaps = sorted(marks[i]-marks[i-1] for i in range(1,len(marks)))
print("engine 36 layers OK, median", round(gaps[len(gaps)//2]*1000), "ms/token")
