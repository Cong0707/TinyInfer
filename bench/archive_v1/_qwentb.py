
import sys, torch, traceback
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
d = r"E:\models\Qwen3.5-0.8B"
tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
ids = tok("你好", return_tensors="pt").input_ids[0].tolist()
eng = Engine(d, tier=OptLevel.BALANCED, device="cuda")
try:
    with torch.inference_mode():
        list(eng.stream_generate(ids, max_new_tokens=4, temperature=0.0))
except Exception:
    traceback.print_exc()
