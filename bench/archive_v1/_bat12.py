
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
print("prompt:", p, "len", len(p))

with torch.inference_mode():
    eng.reset()
    l1 = eng.forward(torch.tensor([p], device="cuda"), 0)
    a = sample_logits(l1[0,-1], 0.0, 1.0, 0)
    print("单序列:", a)

    # 两条完全相同的序列
    eng.reset()
    ids = torch.tensor([p, p], device="cuda")
    lb = eng.forward(ids, 0)
    a0 = sample_logits(lb[0,-1], 0.0, 1.0, 0)
    a1 = sample_logits(lb[1,-1], 0.0, 1.0, 0)
    print("等长批2: row0", a0, " row1", a1, " 一致:", a0==a and a1==a)
    print("  row0 vs 单序列 logits maxdiff:", (lb[0,-1].float()-l1[0,-1].float()).abs().max().item())
