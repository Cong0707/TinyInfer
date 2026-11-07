
import sys, torch, traceback
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer
torch.autograd.set_detect_anomaly(False)

d = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device="cuda", layers=list(range(40)))
p1 = tok("Hello", return_tensors="pt").input_ids[0].tolist()
W = len(p1)
ids = torch.full((2, W), eng.pad_id, dtype=torch.long, device="cuda")
ids[0, :W] = torch.tensor(p1)
ids[1, :W] = torch.tensor(p1)
with torch.inference_mode():
    eng.reset()
    try:
        l = eng.forward(ids, 0)
        print("ok", tuple(l.shape))
    except Exception:
        traceback.print_exc()
