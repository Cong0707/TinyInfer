
import sys, time, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer

d = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
ids = tok("Hello, I am DeepSeek.", return_tensors="pt").input_ids[0].tolist()
eng = Engine(d, tier=OptLevel.BALANCED, device="cuda", layers=list(range(40)))
with torch.inference_mode():
    list(eng.stream_generate(ids, max_new_tokens=3, temperature=0.0))  # warm
    for rnd in range(3):
        marks = []
        for _ in eng.stream_generate(ids, max_new_tokens=12, temperature=0.0):
            marks.append(time.time())
        gaps = sorted(marks[i]-marks[i-1] for i in range(2, len(marks)))
        med = gaps[len(gaps)//2]
        print(f"round{rnd}: {med*1000:.0f} ms/token -> {1/med:.2f} tok/s")
