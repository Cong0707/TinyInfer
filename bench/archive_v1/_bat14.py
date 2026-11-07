
import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from tinyinfer.deepseek.model import Block
from transformers import AutoTokenizer

d = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device="cuda", layers=list(range(40)))
p = tok("Hello", return_tensors="pt").input_ids[0].tolist()
orig = Block.forward
rec = []
def patched(self, x, start_pos, pre_mix, positions=None):
    out = orig(self, x, start_pos, pre_mix, positions=positions)
    rec.append((self.layer_id, out[0].detach().clone()))
    return out
Block.forward = patched
res = {}
with torch.inference_mode():
    for tag, n in (("m1", 1), ("m2", 2)):
        eng.reset(); rec.clear()
        eng.forward(torch.tensor([p]*n, device="cuda"), 0)
        torch.cuda.synchronize()
        res[tag] = [(l, t.clone()) for l, t in rec]
Block.forward = orig
print(f"层数 m1={len(res['m1'])} m2={len(res['m2'])}")
for (l1,a),(l2,b) in zip(res["m1"], res["m2"]):
    dm = (a.float() - b[0:1].float()).abs().max().item()
    if dm > 0:
        print(f"第一个分叉: 层 {l1}  maxdiff={dm:.6f}")
        break
else:
    print("全部一致")
