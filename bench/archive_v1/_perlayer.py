import sys
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
import torch
import tinyinfer.torch_compat  # noqa: F401
from tinyinfer import gdn_patch, qwen_norm
gdn_patch.install(); qwen_norm.install()
from tinyinfer.engine import Engine
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F

MD = r"E:\Models\Qwen3.5-0.8B"
tok = AutoTokenizer.from_pretrained(MD, trust_remote_code=True)
ids = tok("hi", return_tensors="pt").input_ids.to("cuda")

ref = AutoModelForCausalLM.from_pretrained(MD, dtype=torch.bfloat16,
                                           trust_remote_code=True).eval().to("cuda")
# per-layer comparison: capture hidden states from the reference for every layer
states = {}
def mk(i):
    def hook(m, inp, out):
        states[i] = (out[0] if isinstance(out, tuple) else out).detach()
    return hook
hs = [ref.model.layers[i].register_forward_hook(mk(i)) for i in range(24)]
with torch.inference_mode():
    ref(input_ids=ids)
for h in hs: h.remove()
del ref; torch.cuda.empty_cache()

eng = Engine.from_hf(MD, device="cuda", allow_missing_layers=True)
kv = eng.adapter.make_kv(64)
model = eng.adapter.model
with torch.inference_mode():
    pos = torch.arange(0, ids.shape[1], device="cuda")
    h = F.embedding(ids, eng.embed)
    rope = model.position_embeddings(h, pos)
    tp = model.text_positions(pos)
    worst = []
    for i in range(24):
        lay = eng.block(i)
        mask = model.attention_mask(h, tp, cache=kv, layer_id=i)
        out = lay.forward(h, cache=kv, position=tp, position_embeddings=rope,
                          attention_mask=mask)
        h = out[0] if isinstance(out, tuple) else out
        d = float((h[0,-1].float() - states[i][0,-1].float()).abs().max())
        worst.append((d, i))
worst.sort(reverse=True)
print("per-layer hidden diff:", [(i, round(d, 4)) for d, i in worst[:6]], flush=True)
