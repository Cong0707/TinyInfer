
import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer

d = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
ids = tok("Hello, I am DeepSeek.", return_tensors="pt").input_ids[0].tolist()
for tier in (OptLevel.BALANCED,):
    try:
        eng = Engine(d, tier=tier, device="cuda", layers=list(range(40)))
        print(f"[{tier.name}] built (unexpected)")
    except Exception as e:
        print(f"[{tier.name}] REFUSED as designed: {str(e)[:160]}")
