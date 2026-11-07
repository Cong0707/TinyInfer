
import os, sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer

d = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
p = tok("Hello", return_tensors="pt").input_ids[0].tolist()
os.environ["TI_MOE_FP4"] = "0"          # 都走 bf16（稀疏专家走 dequant）
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device="cuda", layers=list(range(40)))
with torch.inference_mode():
    eng.reset(); a = eng.forward(torch.tensor([p], device="cuda"), 0)[0,-1].float()
    eng.reset(); b = eng.forward(torch.tensor([p,p], device="cuda"), 0)[0,-1].float()
print("TI_MOE_FP4=0 全 bf16:  M=1 vs M=2 maxdiff =", (a-b).abs().max().item())
print("M=1 argmax:", int(a.argmax()), " M=2 argmax:", int(b.argmax()))
