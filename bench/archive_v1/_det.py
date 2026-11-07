
import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer

d = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
p = tok("Hello", return_tensors="pt").input_ids[0].tolist()
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device="cuda", layers=list(range(40)))
outs = []
with torch.inference_mode():
    for i in range(2):
        eng.reset()
        outs.append(list(eng.stream_generate(p, max_new_tokens=8, temperature=0.0)))
print("run1:", outs[0])
print("run2:", outs[1])
print("可重复:", outs[0] == outs[1])
print("文本:", repr(tok.decode(outs[0])))
print("显存峰值:", round(torch.cuda.max_memory_allocated()/2**20), "MB")
