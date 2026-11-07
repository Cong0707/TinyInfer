
import os
os.environ["TI_FP8_NEW"] = "1"
import torch, sys
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from tinyinfer.deepseek.model import Block
from transformers import AutoTokenizer

d = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
ids = tok("Hello", return_tensors="pt").input_ids
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device="cuda", layers=list(range(40)))
orig = Block.forward
rec = []
def patched(self, x, start_pos, pre_mix):
    out = orig(self, x, start_pos, pre_mix)
    rec.append((self.layer_id, out[0].detach().clone()))
    return out
Block.forward = patched
res = {}
with torch.inference_mode():
    for use in ("0", "1"):
        os.environ["TI_FP8_NEW"] = use
        rec.clear()
        eng._forward_impl(ids, 0, None)
        torch.cuda.synchronize()
        res[use] = [(lid, t.clone()) for lid, t in rec]
Block.forward = orig
diff_at = None
for (l1, a), (l2, b) in zip(res["0"], res["1"]):
    dm = (a.float()-b.float()).abs().max().item()
    if dm > 0:
        diff_at = (l1, dm); break
print("first diff:", diff_at if diff_at else "none - all 40 layers identical")
