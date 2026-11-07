
import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer

d = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device="cuda", layers=list(range(40)))
p1 = tok("Hello", return_tensors="pt").input_ids[0].tolist()
p2 = tok("The capital of France is", return_tensors="pt").input_ids[0].tolist()
pad = eng.pad_id

with torch.inference_mode():
    eng.reset()
    l1 = eng.forward(torch.tensor([p1], device="cuda"), 0)[0, -1].float()
    eng.reset()
    W = max(len(p1), len(p2))
    ids = torch.full((2, W), pad, dtype=torch.long, device="cuda")
    ids[0, :len(p1)] = torch.tensor(p1); ids[1, :len(p2)] = torch.tensor(p2)
    lb = eng.forward(ids, 0)
    print("prefill 时 A 行 vs 单序列:",
          (lb[0, len(p1)-1].float() - l1).abs().max().item())
    # 第二步：各自送一个新 token，位置分别正确
    eng.reset()
    lb0 = eng.forward(ids, 0)
    nxt = torch.zeros((2, 1), dtype=torch.long, device="cuda")
    nxt[0,0] = 989; nxt[1,0] = 11111
    # A 位置 = len(p1)，B 位置 = len(p2) —— 但 forward 只收一个 start_pos
    l2 = eng.forward(nxt, start_pos=len(p2))
    print("第二步（start_pos=len(p2)）A 行 logits 前3:", l2[0,0,:3].tolist())
    print("说明：start_pos 只能给一个值，两个序列位置不同时 KV 会错位")
