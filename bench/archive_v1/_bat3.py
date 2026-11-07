
import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer

d = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device="cuda", layers=list(range(2)))
a = tok("Hello world", return_tensors="pt").input_ids[0].tolist()
b = tok("The quick brown", return_tensors="pt").input_ids[0].tolist()
pad = tok.eos_token_id if tok.eos_token_id is not None else 0
print("prompt a:", a, " b:", b, " pad:", pad)

with torch.inference_mode():
    eng.reset()
    la = eng.forward(torch.tensor([a], device="cuda"), 0)
    ref = la[0, -1].float()

    W = max(len(a), len(b))
    # 右填充：真实 token 在左，pad 在右
    pa = a + [pad] * (W - len(a))
    pb = b + [pad] * (W - len(b))
    eng.reset()
    lb = eng.forward(torch.tensor([pa, pb], device="cuda"), 0)
    # row0 的最后一个真实 token 在 index len(a)-1
    got = lb[0, len(a) - 1].float()
    print("右填充 row0@len(a)-1 vs 单序列 maxdiff:", (got - ref).abs().max().item())
