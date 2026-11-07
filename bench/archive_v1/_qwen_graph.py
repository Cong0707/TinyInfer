
import sys, time, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer

d = r"E:\models\Qwen3.5-0.8B"
tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
ids = tok("你好", return_tensors="pt").input_ids[0].tolist()

for label, tier in (("BALANCED(graph)", OptLevel.BALANCED),):
    try:
        eng = Engine(d, tier=tier, device="cuda")
        with torch.inference_mode():
            list(eng.stream_generate(ids, max_new_tokens=4, temperature=0.0))
            marks = []
            for _ in eng.stream_generate(ids, max_new_tokens=24, temperature=0.0):
                marks.append(time.time())
        gaps = sorted(marks[i]-marks[i-1] for i in range(4, len(marks)))
        med = gaps[len(gaps)//2]
        g = getattr(eng, "_graph", None)
        print(f"{label}: {med*1000:.2f} ms/token -> {1/med:.1f} tok/s  graph={g is not None}")
    except Exception as e:
        print(f"{label} FAILED: {str(e)[:150]}")
