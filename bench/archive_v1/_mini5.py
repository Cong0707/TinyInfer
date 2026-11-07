
import os
os.environ["TI_FUSE"] = "1"
import torch, sys
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
import tinyinfer.deepseek.model as M
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
    for use in (False, True):
        M._FUSE_HC_PRE = use
        rec.clear()
        eng._forward_impl(ids, 0, None)
        torch.cuda.synchronize()
        res[use] = [(lid, t.clone()) for lid, t in rec]
Block.forward = orig

for (l1, a), (l2, b) in zip(res[False], res[True]):
    dm = (a.float()-b.float()).abs().max().item()
    if dm > 0:
        print(f"layer {l1:2d} maxdiff={dm:.3e}  <-- FIRST DIFF")
        break
else:
    print("all layers identical")
