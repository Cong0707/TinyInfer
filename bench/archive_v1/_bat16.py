
import sys, torch
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
from tinyinfer.engine import Engine
from tinyinfer.tiers import OptLevel
from transformers import AutoTokenizer

d = r"E:\models\DeepSeek-V4.1-Flash-1017"
tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
eng = Engine(d, tier=OptLevel.AGGRESSIVE, device="cuda", layers=list(range(40)))
p = tok("Hello", return_tensors="pt").input_ids[0].tolist()
b0 = eng.block(0)
print("layer0 attn type:", type(b0.attn).__name__)

rec = {}
orig_attn = b0.attn.forward
def patched_attn(x, start_pos=0, positions=None):
    y = orig_attn(x, start_pos, positions=positions)
    rec.setdefault("attn_in", []).append(x.detach().clone())
    rec.setdefault("attn_out", []).append(y.detach().clone())
    return y
b0.attn.forward = patched_attn

orig_hc = b0.hc_mixes
def patched_hc(*a, **k):
    r = orig_hc(*a, **k)
    rec.setdefault("hc", []).append(tuple(t.detach().clone() for t in r))
    return r
b0.hc_mixes = patched_hc

with torch.inference_mode():
    for tag, n in (("m1",1), ("m2",2)):
        eng.reset(); rec.clear()
        eng.forward(torch.tensor([p]*n, device="cuda"), 0)
        torch.cuda.synchronize()
        for key in ("hc","attn_in","attn_out"):
            v = rec.get(key)
            if v is None: continue
            if key == "hc":
                t = v[0][0]
                print(f"{tag} hc[0] {tuple(t.shape)}")
            else:
                t = v[0]
                print(f"{tag} {key} {tuple(t.shape)}")
        globals()["R_"+tag] = {k: [x.clone() if torch.is_tensor(x) else tuple(y.clone() for y in x) for x in v] for k,v in rec.items()}

for key in ("hc","attn_in","attn_out"):
    a = R_m1.get(key); b = R_m2.get(key)
    if not a or not b: continue
    if key == "hc":
        for j in range(len(a[0])):
            dm = (a[0][j].float() - b[0][j][0:1].float()).abs().max().item()
            print(f"  hc[{j}] maxdiff {dm:.3e}")
    else:
        dm = (a[0].float() - b[0][0:1].float()).abs().max().item()
        print(f"  {key} maxdiff {dm:.3e}")
