import sys
sys.path.insert(0, r"E:\Develop\Deepseek\tinyinfer\src")
import torch
import tinyinfer.torch_compat  # noqa: F401
from tinyinfer import gdn_patch, qwen_norm
gdn_patch.install(); qwen_norm.install()
from tinyinfer.engine import Engine
from transformers import AutoModelForCausalLM, AutoTokenizer

MD = r"E:\Models\Qwen3.5-0.8B"
tok = AutoTokenizer.from_pretrained(MD, trust_remote_code=True)
ids = tok("hi", return_tensors="pt").input_ids.to("cuda")
ref = AutoModelForCausalLM.from_pretrained(MD, dtype=torch.bfloat16,
                                           trust_remote_code=True).eval().to("cuda")
cap = {}
hs = [ref.model.layers[i].register_forward_hook(
        (lambda i: (lambda m, a, o: cap.__setitem__(i, (o[0] if isinstance(o,tuple) else o).detach())))(i))
      for i in range(24)]
with torch.inference_mode():
    full = ref(input_ids=ids)
for h in hs: h.remove()

eng = Engine.from_hf(MD, device="cuda", allow_missing_layers=True)
model = eng.adapter.model
# reference final norm applied to the same hidden state the engine produced
last = cap[23]
with torch.inference_mode():
    ref_norm = ref.model.norm(last)
    my_norm = model.final_norm(last)
print("norm diff amax:", float((ref_norm.float()-my_norm.float()).abs().max()), flush=True)
print("norm weights equal:", bool((ref.model.norm.weight.float()-eng.out_norm.float()).abs().max()==0), flush=True)
# and the head
r_logits = full.logits
with torch.inference_mode():
    m_logits = torch.nn.functional.linear(my_norm, eng.head)
d = (r_logits[:, -1].float() - m_logits[:, -1].float()).abs()
print("logits diff after norm fix amax:", float(d.max()), flush=True)
