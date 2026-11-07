
import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
from tinyinfer.nn import sample_logits

d = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device="cuda", layers=list(range(40)))
p = tok("Hello", return_tensors="pt").input_ids[0].tolist()

def run(n_rows, force_kernel):
    import os
    if force_kernel: os.environ["TI_MOE_FP4"] = "1"
    else: os.environ.pop("TI_MOE_FP4", None)
    with torch.inference_mode():
        eng.reset()
        ids = torch.tensor([p]*n_rows, device="cuda")
        l = eng.forward(ids, 0)
        return l[0,-1].float()

import os
a1_k = run(1, True)     # M=1 强制 kernel
a1_b = run(1, False)    # M=1 默认(kernel)
a2_k = run(2, True)     # M=2 强制 kernel -> 走 prefill bf16
print("M=1 kernel vs M=2 强制kernel maxdiff:", (a1_k-a2_k).abs().max().item())
print("M=1 默认 vs M=1 kernel maxdiff:", (a1_b-a1_k).abs().max().item())
