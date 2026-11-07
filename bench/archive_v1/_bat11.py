
import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
from tinyinfer.nn import sample_logits

d = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device="cuda", layers=list(range(40)))
p1 = tok("Hello", return_tensors="pt").input_ids[0].tolist()
p2 = tok("The capital of France is", return_tensors="pt").input_ids[0].tolist()
print("len p1", len(p1), "len p2", len(p2))

with torch.inference_mode():
    eng.reset()
    # 单序列 A 的两步，拿 logits 参考
    l_a1 = eng.forward(torch.tensor([p1], device="cuda"), 0)
    a1 = sample_logits(l_a1[0,-1], 0.0, 1.0, 0)
    l_a2 = eng.forward(torch.tensor([[a1]], device="cuda"), start_pos=len(p1))
    a2 = sample_logits(l_a2[0,-1], 0.0, 1.0, 0)
    print("单序列 A:", a1, a2)

    # 批 2 同样两步
    W = max(len(p1), len(p2))
    pad = eng.pad_id
    ids = torch.full((2, W), pad, dtype=torch.long, device="cuda")
    ids[0,:len(p1)] = torch.tensor(p1); ids[1,:len(p2)] = torch.tensor(p2)
    eng.reset()
    pos0 = torch.tensor([len(p1)-1, len(p2)-1], device="cuda")
    lb1 = eng.forward(ids, start_pos=int(pos0.min()), positions=pos0)
    b_a1 = sample_logits(lb1[0, len(p1)-1], 0.0, 1.0, 0)
    b_b1 = sample_logits(lb1[1, len(p2)-1], 0.0, 1.0, 0)
    print("批2 第1步: A", b_a1, " B", b_b1)
    print("  A 与单序列一致:", b_a1 == a1)
    # 第二步
    nxt = torch.tensor([[b_a1],[b_b1]], device="cuda")
    pos1 = torch.tensor([len(p1), len(p2)], device="cuda")
    lb2 = eng.forward(nxt, start_pos=int(pos1.min()), positions=pos1)
    b_a2 = sample_logits(lb2[0,0], 0.0, 1.0, 0)
    print("批2 第2步 A:", b_a2, " 单序列:", a2, " 一致:", b_a2 == a2)
