
import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer

d = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device="cuda", layers=list(range(2)))
a = tok("Hello world", return_tensors="pt").input_ids[0].tolist()
nxt = tok(" the", return_tensors="pt").input_ids[0].tolist()

with torch.inference_mode():
    # 单序列：prefill + 续一个 token
    eng.reset()
    l1 = eng.forward(torch.tensor([a], device="cuda"), 0)
    l2 = eng.forward(torch.tensor([[nxt[0]]], device="cuda"), start_pos=len(a))
    print("单序列 step2 logits:", tuple(l2.shape))

    # 批 2：同样 prompt，第二个不同
    b = tok("The quick brown", return_tensors="pt").input_ids[0].tolist()
    W = max(len(a), len(b))
    pa = [0]*(W-len(a)) + a
    pb = [0]*(W-len(b)) + b
    eng.reset()
    lb1 = eng.forward(torch.tensor([pa, pb], device="cuda"), 0)
    lb2 = eng.forward(torch.tensor([[nxt[0]], [nxt[0]]], device="cuda"), start_pos=W)
    print("批2 step2 logits:", tuple(lb2.shape))
    d0 = (lb1[0, -1].float() - l1[0, -1].float()).abs().max().item()
    d2 = (lb2[0].float() - l2[0].float()).abs().max().item()
    print("批2 第1步 row0 vs 单序列 maxdiff:", d0)
    print("批2 第2步 row0 vs 单序列 maxdiff:", d2)
