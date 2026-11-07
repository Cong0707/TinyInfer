
import sys, torch, time
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer

d = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device="cuda", layers=list(range(40)))
prompt = tok("The capital of France is", return_tensors="pt").input_ids[0].tolist()
with torch.inference_mode():
    eng.reset()
    out = list(eng.stream_generate(prompt, max_new_tokens=20, temperature=0.0))
print("问题: The capital of France is")
print("回答:", repr(tok.decode(out)))
print()
p2 = tok("1, 2, 3, 4, 5,", return_tensors="pt").input_ids[0].tolist()
with torch.inference_mode():
    eng.reset()
    out2 = list(eng.stream_generate(p2, max_new_tokens=12, temperature=0.0))
print("问题: 1, 2, 3, 4, 5,")
print("回答:", repr(tok.decode(out2)))
