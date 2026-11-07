
import sys, torch, traceback
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer

d = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device="cuda", layers=list(range(4)))
p1 = tok("Hello", return_tensors="pt").input_ids[0].tolist()
p2 = tok("The capital of France is", return_tensors="pt").input_ids[0].tolist()
W = max(len(p1), len(p2))
pad = eng.pad_id
ids = torch.full((2, W), pad, dtype=torch.long, device="cuda")
ids[0, :len(p1)] = torch.tensor(p1)
ids[1, :len(p2)] = torch.tensor(p2)
with torch.inference_mode():
    eng.reset()
    try:
        l = eng.forward(ids, 0)
        print("prefill ok", tuple(l.shape))
        nxt = torch.full((2,1), pad, dtype=torch.long, device="cuda")
        nxt[0,0] = 100; nxt[1,0] = 200
        l2 = eng.forward(nxt, start_pos=0)
        print("step2 ok", tuple(l2.shape))
    except Exception:
        traceback.print_exc()
