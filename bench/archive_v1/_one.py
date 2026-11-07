
import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from tinyinfer.deepseek.model import Block
from transformers import AutoTokenizer

d = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
p = tok("Hello", return_tensors="pt").input_ids[0].tolist()
# 只加载第 0 层：单层权重，200MB 级别
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device="cuda", layers=[0])

orig = Block.forward; rec = []
def patched(self, x, start_pos, pre_mix, positions=None):
    out = orig(self, x, start_pos, pre_mix, positions=positions)
    rec.append(out[0].detach().float().clone())
    return out
Block.forward = patched
with torch.inference_mode():
    res = {}
    for tag, n in (("m1",1), ("m2",2)):
        eng.reset(); rec.clear()
        eng.forward(torch.tensor([p]*n, device="cuda"), 0)
        torch.cuda.synchronize()
        res[tag] = rec[0].clone()
Block.forward = orig
print("layer0 M=1 vs M=2 maxdiff:", (res["m1"] - res["m2"][0:1]).abs().max().item())
print("显存:", round(torch.cuda.max_memory_allocated()/2**20), "MB")
